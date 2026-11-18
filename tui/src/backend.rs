// SPDX-License-Identifier: AGPL-3.0-or-later
//! Process plumbing for the TUI: find the Python package, run its commands,
//! and stream their output into the terminal's log panel.
//!
//! The TUI never reimplements the maths or invents numbers. Every line in the log
//! came from the backing command; every result shown is parsed from the JSON that
//! command printed.

use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::mpsc::{self, Sender};
use std::thread;

use anyhow::{anyhow, Context, Result};
use serde_json::Value;

#[derive(Clone, Debug)]
pub enum JobEvent {
    Line(String),
    Done { code: i32, label: String },
    /// A read-only JSON readout finished off the UI thread.
    Targets(Result<Value, String>),
}

pub struct Repo {
    pub root: PathBuf,
    pub python: PathBuf,
}

impl Repo {
    /// Locate the Python package: `--repo`, then `JDEWITT_REPO`, then an ancestor
    /// search from the current directory, then `~/.jdewitt`.
    pub fn discover(explicit: Option<PathBuf>) -> Result<Repo> {
        let mut candidates: Vec<PathBuf> = Vec::new();
        if let Some(p) = explicit {
            candidates.push(p);
        }
        if let Ok(env) = std::env::var("JDEWITT_REPO") {
            candidates.push(PathBuf::from(env));
        }
        if let Ok(cwd) = std::env::current_dir() {
            let mut cur = Some(cwd.as_path());
            while let Some(d) = cur {
                candidates.push(d.to_path_buf());
                cur = d.parent();
            }
        }
        if let Ok(home) = std::env::var("HOME") {
            candidates.push(PathBuf::from(home).join(".jdewitt"));
        }
        let root = candidates
            .into_iter()
            .find(|d| d.join("jdewitt").join("spanfit.py").exists())
            .ok_or_else(|| anyhow!("no jdewitt package found; pass --repo /path/to/repo"))?;
        let venv = root.join(".venv").join("bin").join("python");
        let python = if venv.exists() {
            venv
        } else {
            PathBuf::from("python3")
        };
        Ok(Repo { root, python })
    }

    pub fn cli<S: AsRef<std::ffi::OsStr>>(&self, args: &[S]) -> std::process::Command {
        let mut c = std::process::Command::new(&self.python);
        c.current_dir(&self.root)
            .args(["-m", "jdewitt.cli"])
            .args(args);
        c
    }

    pub fn spanfit<S: AsRef<std::ffi::OsStr>>(&self, args: &[S]) -> std::process::Command {
        let mut c = std::process::Command::new(&self.python);
        c.current_dir(&self.root)
            .args(["-m", "jdewitt.spanfit"])
            .args(args);
        c
    }

    /// Run a read-only JSON command to completion and parse its reply.
    pub fn run_json(&self, cmd: &mut std::process::Command) -> Result<Value> {
        let out = cmd
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .context("could not start the Python command")?;
        if !out.status.success() {
            let err = String::from_utf8_lossy(&out.stderr);
            let first = err.lines().rev().find(|l| !l.trim().is_empty()).unwrap_or("failed").trim().to_string();
            return Err(anyhow!("command failed: {first}"));
        }
        let text = String::from_utf8_lossy(&out.stdout).to_string();
        let start = text
            .find('{')
            .ok_or_else(|| anyhow!("command printed no JSON: {}", text.trim()))?;
        let end = text.rfind('}').context("no closing brace")?;
        Ok(serde_json::from_str(&text[start..=end])?)
    }

    pub fn cli_json<S: AsRef<std::ffi::OsStr>>(&self, args: &[S]) -> Result<Value> {
        self.run_json(&mut self.cli(args))
    }

    pub fn spanfit_json(&self, args: &[&str]) -> Result<Value> {
        self.run_json(&mut self.spanfit(args))
    }
}

/// Start a long command in the background; its stdout/stderr lines arrive as events.
pub fn start(cmd: &mut std::process::Command, label: &str, tx: &Sender<JobEvent>) -> Result<()> {
    let label = label.to_string();
    let mut child = cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("could not start {label}"))?;
    let stdout = child.stdout.take().context("stdout")?;
    let stderr = child.stderr.take().context("stderr")?;
    for stream in [Box::new(stdout) as Box<dyn std::io::Read + Send>, Box::new(stderr)] {
        let txs = tx.clone();
        thread::spawn(move || {
            let reader = BufReader::new(stream);
            for line in reader.lines() {
                match line {
                    Ok(l) => {
                        let _ = txs.send(JobEvent::Line(l));
                    }
                    Err(_) => break,
                }
            }
        });
    }
    let txs = tx.clone();
    let label2 = label.clone();
    thread::spawn(move || {
        let code = match child.wait() {
            Ok(s) => s.code().unwrap_or(-1),
            Err(_) => -1,
        };
        let _ = txs.send(JobEvent::Done { code, label: label2 });
    });
    Ok(())
}

/// Read `spanfit targets` off the UI thread; the result arrives as an event.
pub fn read_targets(repo: &Repo, path: PathBuf, tx: &mpsc::Sender<JobEvent>) {
    let repo = Repo {
        root: repo.root.clone(),
        python: repo.python.clone(),
    };
    let tx = tx.clone();
    thread::spawn(move || {
        let result = repo
            .spanfit_json(&["targets", "--gguf-in", &path.display().to_string()])
            .map_err(|e| e.to_string());
        let _ = tx.send(JobEvent::Targets(result));
    });
}

/// List the tensor keys inside a `--axes` .pt so the recipe can pick the right one.
pub fn read_axes_keys(repo: &Repo, path: &std::path::Path) -> Result<Vec<String>> {
    let script = "import sys, torch; d = torch.load(sys.argv[1], map_location='cpu', weights_only=False); \
print(','.join(sorted(d.keys())) if isinstance(d, dict) else 'axes')";
    let out = Command::new(&repo.python)
        .current_dir(&repo.root)
        .args(["-c", script])
        .arg(path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .context("could not inspect the axes file")?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        let first = err
            .lines()
            .rev()
            .find(|l| !l.trim().is_empty())
            .unwrap_or("failed")
            .trim()
            .to_string();
        return Err(anyhow!("axes file unreadable: {first}"));
    }
    let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
    Ok(text.split(',').map(|s| s.to_string()).filter(|s| !s.is_empty()).collect())
}
/// Build the right command for a recipe and launch it in the background.
/// `kind` is "cli" or "spanfit"; the caller owns the channel and drains it while drawing.
pub fn start_repo_job(repo: &Repo, kind: &str, args: &[String], label: &str, tx: &Sender<JobEvent>) -> Result<()> {
    let mut cmd = match kind {
        "cli" => repo.cli(args),
        "spanfit" => repo.spanfit(args),
        other => return Err(anyhow!("unknown command kind: {other}")),
    };
    start(&mut cmd, label, tx)
}
