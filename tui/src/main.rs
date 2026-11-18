// SPDX-License-Identifier: AGPL-3.0-or-later
//! jdewitt-tui — one flow, three honest recipes, everything on screen comes from
//! the command that ran. Ratatui for layout, tachyonfx for the transitions.

use std::path::PathBuf;
use std::sync::mpsc;
use std::time::{Duration as StdDuration, Instant};

use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEventKind, MouseEvent, MouseEventKind};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, ListState, Paragraph};
use ratatui::{Frame, Terminal};
use tachyonfx::{fx, Duration, EffectManager, EffectTimer, Interpolation, Motion};

mod app;
mod backend;

use crate::app::{App, Mode, Pick, Screen};
use crate::backend::{start_repo_job, JobEvent, Repo};

fn main() -> Result<()> {
    let mut repo_arg: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        if a == "--repo" {
            repo_arg = args.next().map(PathBuf::from);
        } else if a == "--help" || a == "-h" {
            println!("usage: jdewitt-tui [--repo /path/to/JdeWitt]");
            println!();
            println!("Pick a GGUF on disk, pick a recipe, watch the real command run.");
            println!("Environment: JDEWITT_REPO, JDEWITT_TUI_ROOTS (colon-separated GGUF dirs).");
            return Ok(());
        }
    }
    let repo = Repo::discover(repo_arg)?;
    let mut app = App::new(repo.root.clone());
    app.scan_files();

    let (tx, rx) = mpsc::channel::<JobEvent>();
    enable_raw_mode()?;
    let mut output = std::io::stdout();
    execute!(output, EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(output))?;

    let result = run(&mut terminal, &repo, &tx, &rx, &mut app);

    let mut output = std::io::stdout();
    execute!(output, LeaveAlternateScreen)?;
    disable_raw_mode()?;

    match result {
        Ok(()) => {
            if let Some(code) = app.last_code {
                println!("last command exit: {code}");
            }
            Ok(())
        }
        Err(e) => {
            eprintln!("tui error: {e}");
            Err(e)
        }
    }
}

fn run(
    terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    repo: &Repo,
    tx: &mpsc::Sender<JobEvent>,
    rx: &mpsc::Receiver<JobEvent>,
    app: &mut App,
) -> Result<()> {
    let mut fx_manager: EffectManager<String> = EffectManager::default();
    let mut list_state = ListState::default();
    let mut last = Instant::now();
    let mut dirty = true;

    loop {
        let now = Instant::now();
        let dt = StdDuration::from_secs_f64((now - last).as_secs_f64()).max(StdDuration::from_millis(1));
        last = now;

        while let Ok(ev) = rx.try_recv() {
            match ev {
                JobEvent::Line(line) => app.push_log(line),
                JobEvent::Done { code, label } => {
                    let tint = if code == 0 { Color::Green } else { Color::Red };
                    app.push_log(format!("! {label} exit {code}"));
                    // A `spanfit build` that edited nothing has exited 0 twice tonight,
                    // so the exit code alone is not the claim the user cares about.
                    // Check the file the command was told to write, in the UI, before
                    // calling it done.
                    if let Some(out) = app.out.clone() {
                        match std::fs::metadata(&out) {
                            Ok(m) if m.len() > 0 => {
                                app.push_log(format!(
                                    "artifact: {} bytes at {}",
                                    m.len(),
                                    out.display()
                                ));
                            }
                            Ok(_) => app.push_log(format!(
                                "NO ARTIFACT: {} is zero bytes despite exit 0",
                                out.display()
                            )),
                            Err(_) => app.push_log(format!(
                                "NO ARTIFACT: exit {} but {} was never written",
                                code,
                                out.display()
                            )),
                        }
                    }
                    app.running = None;
                    app.last_code = Some(code);
                    fx_manager.add_effect(fx::fade_from_fg(
                        tint,
                        EffectTimer::from_ms(900, Interpolation::QuadOut),
                    ));
                }
                JobEvent::Targets(result) => {
                    app.targets_busy = false;
                    match result {
                        Ok(v) => {
                            app.targets = Some(v);
                            app.targets_err = None;
                        }
                        Err(e) => {
                            app.targets = None;
                            app.targets_err = Some(e);
                        }
                    }
                    fx_flash(&mut fx_manager, "targets");
                }
            }
            dirty = true;
        }

        if event::poll(StdDuration::from_millis(10))? {
            match event::read()? {
                Event::Key(key) if key.kind == KeyEventKind::Press => {
                    if handle_key(app, repo, tx, &mut list_state, &mut fx_manager, key.code) {
                        break;
                    }
                    dirty = true;
                }
                Event::Mouse(MouseEvent {
                    kind: MouseEventKind::ScrollUp,
                    ..
                }) => {
                    scroll(app, -1);
                    list_state.select(Some(app.cursor));
                    dirty = true;
                }
                Event::Mouse(MouseEvent {
                    kind: MouseEventKind::ScrollDown,
                    ..
                }) => {
                    scroll(app, 1);
                    list_state.select(Some(app.cursor));
                    dirty = true;
                }
                _ => {}
            }
        }

        if dirty || app.targets_busy || fx_manager.is_running() {
            terminal.draw(|frame| draw(frame, app, &mut fx_manager, &mut list_state, dt))?;
            dirty = false;
        }
    }
    Ok(())
}

fn fx_flash(manager: &mut EffectManager<String>, key: &str) {
    let effect = manager.unique(key, fx::dissolve(EffectTimer::from_ms(450, Interpolation::QuadOut)));
    manager.add_effect(effect);
}

fn screen_name(s: &Screen) -> &'static str {
    match s {
        Screen::Library => "library",
        Screen::Recipe => "recipe",
        Screen::Run => "run",
    }
}

/// Test/debug aid: `JDEWITT_TUI_DUMP=/tmp/x.json jdewitt-tui`, press i.
/// Writes exactly what the UI is holding, so a headless drive can assert on state.
fn dump(app: &App) {
    if let Ok(path) = std::env::var("JDEWITT_TUI_DUMP") {
        let tail: Vec<String> = app.log.iter().rev().take(5).cloned().collect();
        let json = format!(
            "{{\"screen\":\"{}\",\"cursor\":{},\"mode\":\"{:?}\",\"base\":{},\"ablated\":{},\"axes\":{},\"axes_key\":{},\"out\":{},\"cursor_path\":{},\"config\":{},\"targets\":{},\"targets_err\":{},\"missing\":{},\"command\":{:?},\"running\":{},\"last_code\":{},\"log_tail\":{:?}}}",
            screen_name(&app.screen),
            app.cursor,
            app.mode,
            opt_str(&app.base),
            opt_str(&app.ablated),
            opt_str(&app.axes),
            opt_str(&app.axes_key),
            opt_str(&app.out),
            opt_str(&app.cursor_path()),
            opt_str(&app.auto_config),
            app.targets.is_some(),
            opt_str(&app.targets_err),
            opt_str(&app.missing_slot()),
            app.command(),
            opt_str(&app.running),
            match app.last_code {
                Some(c) => c.to_string(),
                None => "null".to_string(),
            },
            tail,
        );
        let _ = std::fs::write(path, json);
    }
}

fn opt_str<T: AsRef<std::ffi::OsStr>>(v: &Option<T>) -> String {
    match v {
        Some(p) => format!("\"{}\"", p.as_ref().to_string_lossy().replace('"', "\\\"")),
        None => "null".to_string(),
    }
}

fn handle_key(
    app: &mut App,
    repo: &Repo,
    tx: &mpsc::Sender<JobEvent>,
    list_state: &mut ListState,
    fx_manager: &mut EffectManager<String>,
    code: KeyCode,
) -> bool {
    if code == KeyCode::Char('i') {
        dump(app);
        return false;
    }
    match app.screen {
        Screen::Library => match code {
            KeyCode::Char('q') | KeyCode::Esc => return true,
            KeyCode::Up | KeyCode::Char('k') => {
                scroll(app, -1);
                list_state.select(Some(app.cursor));
            }
            KeyCode::Down | KeyCode::Char('j') => {
                scroll(app, 1);
                list_state.select(Some(app.cursor));
            }
            KeyCode::Enter => {
                if let Some(p) = app.selected_path() {
                    if app.pick.is_some() {
                        let was_axes = matches!(app.pick, Some(Pick::Axes));
                        app.accept_pick(p.clone());
                        if was_axes {
                            match backend::read_axes_keys(repo, &p) {
                                Ok(keys) => {
                                    app.axes_keys = keys.clone();
                                    app.axes_key = Some(pick_default_key(&keys));
                                }
                                Err(e) => app.error = Some(e.to_string()),
                            }
                        }
                        app.screen = Screen::Recipe;
                        fx_flash(fx_manager, "screen");
                    } else if p.extension().and_then(|e| e.to_str()) == Some("pt") {
                        app.pick = Some(Pick::Axes);
                        app.accept_pick(p);
                        app.screen = Screen::Recipe;
                        fx_flash(fx_manager, "screen");
                    } else {
                        app.selected = Some(p.clone());
                        // A GGUF picked here is the base for the current recipe.
                        match app.mode {
                            Mode::DeltaPack | Mode::Direction => {
                                app.base = Some(p.clone());
                                if app.mode == Mode::Direction {
                                    app.out = app.default_out();
                                }
                            }
                            Mode::Auto => app.auto_config = Some(p.clone()),
                        }
                        inspect_targets(repo, app, &p, tx);
                        app.screen = Screen::Recipe;
                        fx_flash(fx_manager, "screen");
                    }
                }
            }
            KeyCode::Char('r') => app.scan_files(),
            KeyCode::Tab => {
                app.screen = Screen::Recipe;
                fx_flash(fx_manager, "screen");
            }
            _ => {}
        },
        Screen::Recipe => match code {
            KeyCode::Esc => {
                app.screen = Screen::Library;
                app.pick = None;
                fx_flash(fx_manager, "screen");
            }
            KeyCode::Char('q') => return true,
            KeyCode::Tab | KeyCode::Right => {
                app.mode = match app.mode {
                    Mode::DeltaPack => Mode::Direction,
                    Mode::Direction => Mode::Auto,
                    Mode::Auto => Mode::DeltaPack,
                };
                app.pick = None;
                app.sync_output_path();
                fx_flash(fx_manager, "mode");
            }
            KeyCode::Char('1') => {
                app.mode = Mode::DeltaPack;
                app.error = None;
            }
            KeyCode::Char('2') => {
                app.mode = Mode::Direction;
                app.error = None;
                app.sync_output_path();
            }
            KeyCode::Char('3') => {
                app.mode = Mode::Auto;
                app.error = None;
            }
            KeyCode::Char('k') => {
                if app.axes_keys.len() > 1 {
                    let cur = app.axes_key.clone();
                    let idx = app
                        .axes_keys
                        .iter()
                        .position(|k| Some(k) == cur.as_ref())
                        .map(|i| (i + 1) % app.axes_keys.len())
                        .unwrap_or(0);
                    app.axes_key = Some(app.axes_keys[idx].clone());
                }
            }
            KeyCode::Char('b') | KeyCode::Char('B') => {
                if let Some(slot) = app.missing_slot() {
                    app.error = Some(format!("need the {slot} file"));
                } else {
                    launch(repo, app, tx, fx_manager);
                }
            }
            KeyCode::Enter => {
                let slot = app.missing_slot();
                app.pick = Some(match app.mode {
                    Mode::Auto => Pick::Main,
                    Mode::DeltaPack if slot == Some("ablated") => Pick::Ablated,
                    Mode::DeltaPack => Pick::Main,
                    Mode::Direction if slot == Some("axes") => Pick::Axes,
                    Mode::Direction => Pick::Main,
                });
                app.screen = Screen::Library;
                fx_flash(fx_manager, "pick");
            }
            _ => {}
        },
        Screen::Run => match code {
            KeyCode::Esc => {
                app.screen = Screen::Recipe;
                fx_flash(fx_manager, "screen");
            }
            KeyCode::Char('q') => return true,
            _ => {}
        },
    }
    false
}

/// Which key to load from a multi-key axes file: the shared axis when it exists,
/// since that is the shape the fitted recipes produce.
fn pick_default_key(keys: &[String]) -> String {
    for pref in ["shared", "axes", "per_layer", "pl"] {
        if keys.iter().any(|k| k == pref) {
            return pref.to_string();
        }
    }
    keys.first().cloned().unwrap_or_else(|| "axes".to_string())
}

fn inspect_targets(repo: &Repo, app: &mut App, path: &std::path::Path, tx: &mpsc::Sender<JobEvent>) {
    app.targets = None;
    app.targets_err = None;
    app.targets_busy = true;
    backend::read_targets(repo, path.to_path_buf(), tx);
}

fn source_keys(app: &App) -> Vec<String> {
    let mut keys = Vec::new();
    if let Some(t) = &app.targets {
        if let Some(line) = t.get("build_line").and_then(|v| v.as_str()) {
            let mut it = line.split_whitespace();
            while let Some(tok) = it.next() {
                if tok == "--source-key" {
                    if let Some(k) = it.next() {
                        keys.push(k.to_string());
                    }
                }
            }
        }
        if keys.is_empty() {
            if let Some(by) = t.get("by_source_key").and_then(|v| v.as_object()) {
                for k in by.keys() {
                    keys.push(k.clone());
                }
            }
        }
    }
    keys
}

fn launch(
    repo: &Repo,
    app: &mut App,
    tx: &mpsc::Sender<JobEvent>,
    fx_manager: &mut EffectManager<String>,
) {
    let parts = app.command();
    if parts.len() < 2 {
        return;
    }
    let kind = parts[0].clone();
    let mut args: Vec<String> = parts[1..].to_vec();
    if app.mode == Mode::Direction && app.out.is_none() {
        app.out = app.default_out();
    }
    if app.mode == Mode::Direction && !args.contains(&"--out".to_string()) {
        if let Some(o) = &app.out {
            args.push("--out".to_string());
            args.push(o.display().to_string());
        }
    }
    // Fill the source keys the model actually has; the tool refuses to guess them.
    if app.mode == Mode::Direction && !args.contains(&"--source-key".to_string()) {
        for k in source_keys(app) {
            args.push("--source-key".to_string());
            args.push(k);
        }
    }
    let label = format!("{} {}", kind, args.join(" "));
    app.log.clear();
    app.push_log(format!("$ {label}"));
    app.screen = Screen::Run;
    app.running = Some(label.clone());
    fx_manager.add_effect(fx::slide_in(
        Motion::UpToDown,
        24,
        6,
        Color::DarkGray,
        EffectTimer::from_ms(600, Interpolation::QuadOut),
    ));
    if let Err(e) = start_repo_job(repo, &kind, &args, &label, tx) {
        app.push_log(format!("! {e}"));
        app.running = None;
    }
}

fn scroll(app: &mut App, delta: i32) {
    let len = app.files.len() as i32;
    if len == 0 {
        return;
    }
    let next = ((app.cursor as i32) + delta).clamp(0, len - 1);
    app.cursor = next as usize;
}

fn draw(
    frame: &mut Frame,
    app: &mut App,
    fx_manager: &mut EffectManager<String>,
    list_state: &mut ListState,
    dt: StdDuration,
) {
    let area = frame.area();
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(8),
            Constraint::Length(3),
        ])
        .split(area);

    let picked = app
        .selected
        .as_ref()
        .map(|p| App::short(p))
        .unwrap_or_else(|| "no file picked".to_string());
    let status = match &app.running {
        Some(_) => "RUNNING".to_string(),
        None => match app.last_code {
            Some(0) => "last ok".to_string(),
            Some(c) => format!("last exit {c}"),
            None => "idle".to_string(),
        },
    };
    let status_color = if app.running.is_some() {
        Color::Yellow
    } else {
        Color::DarkGray
    };
    let header = Paragraph::new(Line::from(vec![
        Span::styled(
            " jdewitt ",
            Style::default()
                .fg(Color::Black)
                .bg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(format!(" {}   ", picked)),
        Span::styled(
            format!(" {status} "),
            Style::default().fg(status_color),
        ),
        Span::styled(
            format!(" {}", app.repo_root.display()),
            Style::default().fg(Color::DarkGray),
        ),
    ]))
    .block(
        Block::default()
            .borders(Borders::ALL)
            .title("one-shot refusal removal — on the quant you already have"),
    );
    frame.render_widget(header, rows[0]);

    let body = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(58), Constraint::Percentage(42)])
        .split(rows[1]);

    match app.screen {
        Screen::Library => {
            let title = if app.pick.is_some() {
                match app.pick {
                    Some(Pick::Ablated) => "pick the ablated twin (Enter)".to_string(),
                    Some(Pick::Axes) => "pick the direction pack .pt (Enter)".to_string(),
                    Some(Pick::Main) => format!("pick the {} file (Enter)", app.mode.slot()),
                    Some(Pick::OutDir) => "pick the output location (Enter)".to_string(),
                    None => "pick a file (Enter)".to_string(),
                }
            } else {
                "GGUF files on this machine (j/k, Enter, r rescan)".to_string()
            };
            let items: Vec<ListItem> = app
                .files
                .iter()
                .map(|g| {
                    ListItem::new(Line::from(vec![
                        Span::raw(App::short(&g.path)),
                        Span::styled(
                            format!("  {}", App::human_size(g.size)),
                            Style::default().fg(Color::DarkGray),
                        ),
                    ]))
                })
                .collect();
            if app.files.is_empty() {
                let msg = Paragraph::new(
                    "no .gguf files found.\nSet JDEWITT_TUI_ROOTS to the directories that hold them.",
                )
                .block(Block::default().borders(Borders::ALL).title(title));
                frame.render_widget(msg, body[0]);
            } else {
                let list = List::new(items)
                    .block(Block::default().borders(Borders::ALL).title(title))
                    .highlight_style(
                        Style::default()
                            .bg(Color::DarkGray)
                            .add_modifier(Modifier::BOLD),
                    )
                    .highlight_symbol(">");
                list_state.select(Some(app.cursor));
                frame.render_widget(list, body[0]);
            }
            let note = Paragraph::new(vec![
                Line::from(Span::styled(
                    "the list is a file listing, not a claim about the models",
                    Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC),
                )),
            ])
            .block(Block::default().borders(Borders::ALL).title("note"));
            frame.render_widget(note, body[1]);
        }
        Screen::Recipe => {
            let mut lines: Vec<Line> = vec![Line::from(Span::styled(
                "recipes (Tab cycles, 1/2/3 jump, Enter fills a slot, b runs)\n",
                Style::default().add_modifier(Modifier::BOLD),
            ))];
            for (i, m) in [Mode::DeltaPack, Mode::Direction, Mode::Auto].iter().enumerate() {
                let mark = if *m == app.mode { ">" } else { " " };
                let style = if *m == app.mode {
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(Color::DarkGray)
                };
                lines.push(Line::from(Span::styled(
                    format!("{} {}. {}", mark, i + 1, m.title()),
                    style,
                )));
            }
            lines.push(Line::from(""));
            let slots = match app.mode {
                Mode::DeltaPack => vec![("base", &app.base), ("ablated", &app.ablated)],
                Mode::Direction => vec![("base", &app.base), ("axes", &app.axes)],
                Mode::Auto => vec![("config", &app.auto_config)],
            };
            for (name, slot) in slots {
                let val = slot
                    .as_ref()
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|| "— not set —".to_string());
                let style = if slot.is_some() {
                    Style::default().fg(Color::White)
                } else {
                    Style::default().fg(Color::Red)
                };
                lines.push(Line::from(vec![
                    Span::styled(format!("  {name:>8}: "), Style::default().fg(Color::DarkGray)),
                    Span::styled(val, style),
                ]));
            }
            if app.mode == Mode::Direction && app.axes_key.is_some() {
                lines.push(Line::from(vec![
                    Span::styled("  axes key: ", Style::default().fg(Color::DarkGray)),
                    Span::styled(
                        format!(
                            "{} (k cycles {})",
                            app.axes_key.clone().unwrap_or_default(),
                            app.axes_keys.join(", ")
                        ),
                        Style::default().fg(Color::White),
                    ),
                ]));
            }
            lines.push(Line::from(""));
            lines.push(Line::from(vec![
                Span::styled("command: ", Style::default().fg(Color::DarkGray)),
                Span::styled(app.command().join(" "), Style::default().fg(Color::Gray)),
            ]));
            if let Some(err) = &app.error {
                lines.push(Line::from(Span::styled(
                    format!("! {err}"),
                    Style::default().fg(Color::Red),
                )));
            }
            let bodyw = Paragraph::new(lines)
                .block(Block::default().borders(Borders::ALL).title("step 1 — what you have"));
            frame.render_widget(bodyw, body[0]);

            let info_lines: Vec<Line> = match (&app.targets, &app.targets_err) {
                (Some(t), _) => {
                    let mut v = Vec::new();
                    for k in ["architecture", "layers", "dims", "matched"] {
                        if let Some(x) = t.get(k) {
                            v.push(Line::from(format!("{k}: {x}")));
                        }
                    }
                    for k in source_keys(app) {
                        v.push(Line::from(Span::styled(
                            format!("  source-key {k}"),
                            Style::default().fg(Color::DarkGray),
                        )));
                    }
                    if let Some(tl) = t.get("tensors_loaded") {
                        v.push(Line::from(format!("tensors_loaded: {tl}")));
                    }
                    v.push(Line::from(""));
                    v.push(Line::from(Span::styled(
                        "this is the readout from `spanfit targets`, not the TUI's opinion",
                        Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC),
                    )));
                    v
                }
                (None, Some(e)) => vec![
                    Line::from(Span::styled(
                        "targets readout failed",
                        Style::default().fg(Color::Red),
                    )),
                    Line::from(Span::styled(e.clone(), Style::default().fg(Color::DarkGray))),
                ],
                (None, None) if app.targets_busy => vec![Line::from(Span::styled(
                    "reading `spanfit targets` — this reads the GGUF header, it takes a few seconds",
                    Style::default().fg(Color::Yellow),
                ))],
                (None, None) => vec![Line::from(
                    "pick a GGUF to see what the tool can fit on it",
                )],
            };
            let info = Paragraph::new(info_lines)
                .block(Block::default().borders(Borders::ALL).title("step 2 — what the tool sees"));
            frame.render_widget(info, body[1]);
        }
        Screen::Run => {
            // No scroll widget in this Paragraph: show the tail that fits.
            let height = body[0].height.saturating_sub(2) as usize;
            let start = app.log.len().saturating_sub(height);
            let text: Vec<Line> = app.log[start..]
                .iter()
                .map(|l| {
                    let style = if l.starts_with('$') {
                        Style::default().fg(Color::Cyan)
                    } else if l.starts_with('!') {
                        Style::default().fg(Color::Red)
                    } else if l.contains("exit") {
                        Style::default().fg(Color::Yellow)
                    } else {
                        Style::default()
                    };
                    Line::from(Span::styled(l.clone(), style))
                })
                .collect();
            let log = Paragraph::new(text)
                .block(Block::default().borders(Borders::ALL).title(format!(
                    "running — {}",
                    app.running.clone().unwrap_or_else(|| "finished".to_string())
                )));
            frame.render_widget(log, body[0]);

            let mut summary: Vec<Line> = Vec::new();
            for line in app.log.iter().rev() {
                if line.contains("edited")
                    || line.contains("tensors")
                    || line.contains("exit")
                    || line.contains("error")
                {
                    summary.push(Line::from(line.clone()));
                    if summary.len() >= 10 {
                        break;
                    }
                }
            }
            summary.push(Line::from(Span::styled(
                "the command owns the numbers — the TUI computes none of them",
                Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC),
            )));
            let side = Paragraph::new(summary)
                .block(Block::default().borders(Borders::ALL).title("what it reported"));
            frame.render_widget(side, body[1]);
        }
    }

    let keys = match app.screen {
        Screen::Library => "j/k move   Enter pick   r rescan   Tab recipes   q quit",
        Screen::Recipe => "1/2/3 recipe   Tab next   Enter fill slot   b run   Esc back   q quit",
        Screen::Run => "Esc back to recipe   q quit",
    };
    let footer = Paragraph::new(Span::styled(format!("  {keys}"), Style::default().fg(Color::DarkGray)))
        .block(Block::default().borders(Borders::ALL));
    frame.render_widget(footer, rows[2]);

    fx_manager.process_effects(Duration::from_millis(dt.as_millis() as u32), frame.buffer_mut(), area);
}
