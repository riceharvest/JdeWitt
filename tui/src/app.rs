// SPDX-License-Identifier: AGPL-3.0-or-later
//! App state and actions. The state machine knows what the user has already
//! chosen; it never computes a result of its own.

use std::path::{Path, PathBuf};
use std::time::SystemTime;

use serde_json::Value;

/// The three things the user can already have on disk. The TUI offers exactly these
/// and refuses to promise anything else.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    /// Two GGUFs of the same base, one published ablated: show which tensors moved.
    /// Read-only, no GPU, no HF weights — the honest first step.
    DeltaPack,
    /// One GGUF plus a fitted direction pack (`--axes`), spanfit build.
    Direction,
    /// A full `auto` search under a config the user wrote.
    Auto,
}

impl Mode {
    pub fn title(self) -> &'static str {
        match self {
            Mode::DeltaPack => "inspect — base + a published ablated build, tensor by tensor (read-only, no GPU)",
            Mode::Direction => "direction — base + a fitted .pt direction pack",
            Mode::Auto => "auto — full evidence search under a config file",
        }
    }

    pub fn slot(self) -> &'static str {
        match self {
            Mode::DeltaPack => "base",
            Mode::Direction => "base",
            Mode::Auto => "config",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Pick {
    /// The main file for the current mode (base quant, or the auto config).
    Main,
    /// The ablated twin, delta mode only.
    Ablated,
    /// Direction pack (.pt), direction mode only.
    Axes,
    /// Where to write the result.
    OutDir,
}

pub struct Gguf {
    pub path: PathBuf,
    pub size: u64,
    pub modified: SystemTime,
}

pub enum Screen {
    Library,
    Recipe,
    Run,
}

pub struct App {
    pub repo_root: PathBuf,
    pub screen: Screen,
    pub files: Vec<Gguf>,
    pub cursor: usize,
    pub selected: Option<PathBuf>,
    pub targets: Option<Value>,
    pub targets_err: Option<String>,
    pub targets_busy: bool,
    pub mode: Mode,
    pub pick: Option<Pick>,
    pub base: Option<PathBuf>,
    pub ablated: Option<PathBuf>,
    pub axes: Option<PathBuf>,
    pub out: Option<PathBuf>,
    pub axes_key: Option<String>,
    pub axes_keys: Vec<String>,
    pub auto_config: Option<PathBuf>,
    pub log: Vec<String>,
    pub running: Option<String>,
    pub last_code: Option<i32>,
    pub error: Option<String>,
}

impl App {
    pub fn new(repo_root: PathBuf) -> App {
        App {
            repo_root,
            screen: Screen::Library,
            files: Vec::new(),
            cursor: 0,
            selected: None,
            targets: None,
            targets_err: None,
            targets_busy: false,
            mode: Mode::DeltaPack,
            pick: None,
            base: None,
            ablated: None,
            axes: None,
            out: None,
            axes_key: None,
            axes_keys: Vec::new(),
            auto_config: None,
            log: Vec::new(),
            running: None,
            last_code: None,
            error: None,
        }
    }

    pub fn scan_files(&mut self) {
        let roots: Vec<PathBuf> = match std::env::var("JDEWITT_TUI_ROOTS") {
            Ok(list) => list.split(':').map(PathBuf::from).collect(),
            Err(_) => {
                let home = std::env::var("HOME").unwrap_or_default();
                vec![
                    PathBuf::from(home).join("models"),
                    PathBuf::from("/mnt/ssd/hermes/models"),
                    PathBuf::from("/mnt/ssd/models"),
                    std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
                ]
            }
        };
        let mut found: Vec<Gguf> = Vec::new();
        for root in roots {
            if let Ok(entries) = std::fs::read_dir(&root) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
                    if !matches!(ext, "gguf" | "pt") {
                        continue;
                    }
                    if let Ok(meta) = path.metadata() {
                            found.push(Gguf {
                                path,
                                size: meta.len(),
                                modified: meta.modified().unwrap_or(SystemTime::UNIX_EPOCH),
                            });
                        }
                    }
                }
        }
        found.sort_by(|a, b| b.modified.cmp(&a.modified));
        found.truncate(250);
        self.files = found;
        if self.cursor >= self.files.len() {
            self.cursor = 0;
        }
    }

    pub fn cursor_path(&self) -> Option<PathBuf> {
        self.files.get(self.cursor).map(|g| g.path.clone())
    }

    pub fn selected_path(&self) -> Option<PathBuf> {
        self.files.get(self.cursor).map(|g| g.path.clone()).or(self.selected.clone())
    }

    pub fn default_out(&self) -> Option<PathBuf> {
        let base = self.base.as_ref()?;
        let stem = base.file_stem()?.to_string_lossy().to_string();
        Some(base.with_file_name(format!("{stem}-jdewitt.gguf")))
    }

    /// Assign the file being picked, and move to the next needed slot.
    pub fn accept_pick(&mut self, path: PathBuf) {
        let pick = self.pick.take().unwrap_or(Pick::Main);
        match pick {
            Pick::Main => match self.mode {
                Mode::DeltaPack | Mode::Direction => {
                    self.base = Some(path);
                    if self.mode == Mode::Direction && self.out.is_none() {
                        self.out = self.default_out();
                    }
                }
                Mode::Auto => self.auto_config = Some(path),
            },
            Pick::Ablated => self.ablated = Some(path),
            Pick::Axes => self.axes = Some(path),
            Pick::OutDir => self.out = Some(path),
        }
        self.error = None;
    }

    /// The command line the current recipe will run, as it would be shown to a user.
    pub fn command(&self) -> Vec<String> {
        match self.mode {
            Mode::DeltaPack => {
                let mut c = vec!["cli".to_string(), "gguf-tensor-diff".to_string()];
                if let Some(b) = &self.base {
                    c.push(b.display().to_string());
                }
                if let Some(a) = &self.ablated {
                    c.push(a.display().to_string());
                }
                c
            }
            Mode::Direction => {
                let mut c = vec!["spanfit".to_string(), "build".to_string()];
                if let Some(b) = &self.base {
                    c.push("--gguf-in".to_string());
                    c.push(b.display().to_string());
                }
                if let Some(a) = &self.axes {
                    c.push("--axes".to_string());
                    c.push(a.display().to_string());
                }
                if let Some(k) = &self.axes_key {
                    c.push("--axes-key".to_string());
                    c.push(k.clone());
                }
                if let Some(o) = &self.out {
                    c.push("--out".to_string());
                    c.push(o.display().to_string());
                }
                if let Some(line) = self
                    .targets
                    .as_ref()
                    .and_then(|t| t.get("build_line"))
                    .and_then(|v| v.as_str())
                {
                    c.extend(line.split_whitespace().map(|s| s.to_string()));
                }
                c
            }
            Mode::Auto => {
                let mut c = vec!["cli".to_string(), "auto".to_string()];
                if let Some(a) = &self.auto_config {
                    c.push(a.display().to_string());
                }
                c
            }
        }
    }

    pub fn missing_slot(&self) -> Option<&'static str> {
        let need = |slot: &Option<PathBuf>| slot.is_none();
        match self.mode {
            Mode::DeltaPack => {
                if need(&self.base) {
                    Some("base")
                } else if need(&self.ablated) {
                    Some("ablated")
                } else {
                    None
                }
            }
            Mode::Direction => {
                if need(&self.base) {
                    Some("base")
                } else if need(&self.axes) {
                    Some("axes")
                } else {
                    None
                }
            }
            Mode::Auto => {
                if need(&self.auto_config) {
                    Some("config")
                } else {
                    None
                }
            }
        }
    }

    pub fn push_log(&mut self, line: String) {
        self.log.push(line);
        if self.log.len() > 600 {
            let excess = self.log.len() - 600;
            self.log.drain(0..excess);
        }
    }

    pub fn human_size(n: u64) -> String {
        let kb = n as f64 / 1024.0;
        let mb = kb / 1024.0;
        let gb = mb / 1024.0;
        if gb >= 1.0 {
            format!("{gb:.1} GB")
        } else if mb >= 1.0 {
            format!("{mb:.0} MB")
        } else {
            format!("{kb:.0} KB")
        }
    }

    pub fn short(p: &Path) -> String {
        p.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| p.display().to_string())
    }
}
