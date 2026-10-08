//! Shared progress-bar styling for downloads and uploads.
//!
//! Both operations show the transferred amount against the total, a percentage,
//! the current rate, the elapsed time, and the estimated time remaining. A bar
//! whose total is not known yet (the single-file download probe) starts as a
//! spinner and is upgraded with [`bar_style`] once the size arrives.
//!
//! The bar is a `{wide_bar}`, so it only keeps its width when the rest of the
//! line fits. A long repository id or backend scope would otherwise squeeze it
//! to zero columns, so messages are trimmed to leave [`MIN_BAR_WIDTH`] columns.

use console::{Term, measure_text_width};
use indicatif::{ProgressBar, ProgressStyle};
use std::time::Duration;

/// Template for a bar whose total is known.
const BAR_TEMPLATE: &str = "{spinner:.cyan} {msg} [{wide_bar:.cyan/blue}] \
{decimal_bytes}/{decimal_total_bytes} ({percent}%) • {decimal_bytes_per_sec} • \
elapsed {elapsed_precise} • left {eta_precise}";

/// Template for a spinner whose total is not known yet.
const SPINNER_TEMPLATE: &str = "{spinner:.cyan} {msg} • {decimal_bytes} • {decimal_bytes_per_sec}";

/// Width of everything in [`BAR_TEMPLATE`] except the message and the bar, for
/// typical values (sizes in the `999.99 MB` range, times up to two digits of
/// hours). Extremely large values may borrow a few columns from the bar.
const FIXED_WIDTH: usize = 78;

/// Smallest bar kept visible by trimming a long message.
const MIN_BAR_WIDTH: usize = 16;

/// Upper bound for the message, so wide terminals do not show a wall of text.
const MAX_MESSAGE_WIDTH: usize = 64;

/// Style for a bar whose total is known.
pub(crate) fn bar_style() -> ProgressStyle {
    ProgressStyle::with_template(BAR_TEMPLATE)
        .expect("static bar template")
        .progress_chars("━━─")
}

/// Style for a spinner whose total is not known yet.
pub(crate) fn spinner_style() -> ProgressStyle {
    ProgressStyle::with_template(SPINNER_TEMPLATE).expect("static spinner template")
}

/// Create a progress bar, or a hidden one when `enabled` is false.
///
/// `total` is the number of bytes expected. `None` leaves the length unset so
/// the bar shows a spinner until the caller calls [`ProgressBar::set_length`].
pub(crate) fn bar(enabled: bool, message: impl Into<String>, total: Option<u64>) -> ProgressBar {
    if !enabled {
        return ProgressBar::hidden();
    }
    let bar = total.map_or_else(ProgressBar::no_length, ProgressBar::new);
    bar.set_style(if total.is_some() {
        bar_style()
    } else {
        spinner_style()
    });
    set_message(&bar, message);
    bar.enable_steady_tick(Duration::from_millis(100));
    bar
}

/// Replace the bar's message, trimmed to stay on one line with the bar.
pub(crate) fn set_message(bar: &ProgressBar, message: impl Into<String>) {
    bar.set_message(fit_message(&message.into(), message_budget()));
}

/// Trim `message` to `budget` columns, marking the cut with an ellipsis.
fn fit_message(message: &str, budget: usize) -> String {
    if measure_text_width(message) <= budget {
        return message.to_owned();
    }
    if budget == 0 {
        return String::new();
    }
    let mut fitted = String::new();
    let mut used = 0;
    for character in message.chars() {
        let width = measure_text_width(&character.to_string());
        if used + width > budget - 1 {
            break;
        }
        fitted.push(character);
        used += width;
    }
    fitted.push('…');
    fitted
}

/// Columns left for the message after the fixed template fields and a minimum
/// bar, capped for very wide terminals.
fn message_budget() -> usize {
    Term::stderr()
        .size_checked()
        .map_or(MAX_MESSAGE_WIDTH, |(_, columns)| {
            usize::from(columns)
                .saturating_sub(FIXED_WIDTH + MIN_BAR_WIDTH)
                .min(MAX_MESSAGE_WIDTH)
        })
}

#[cfg(test)]
mod tests {
    use super::{BAR_TEMPLATE, SPINNER_TEMPLATE, bar, fit_message};

    #[test]
    fn bar_template_exposes_total_percent_elapsed_and_eta() {
        for field in [
            "{decimal_total_bytes}",
            "{percent}",
            "{elapsed_precise}",
            "{eta_precise}",
        ] {
            assert!(BAR_TEMPLATE.contains(field), "missing {field}");
        }
    }

    #[test]
    fn spinner_template_has_no_total() {
        assert!(SPINNER_TEMPLATE.contains("{decimal_bytes}"));
        assert!(!SPINNER_TEMPLATE.contains("{decimal_total_bytes}"));
    }

    #[test]
    fn known_total_sets_the_length() {
        let bar = bar(true, "msg".to_owned(), Some(2048));
        assert_eq!(bar.length(), Some(2048));
    }

    #[test]
    fn unknown_total_leaves_the_length_unset() {
        let bar = bar(true, "msg".to_owned(), None);
        assert_eq!(bar.length(), None);
    }

    #[test]
    fn disabled_progress_is_hidden() {
        let bar = bar(false, "msg".to_owned(), Some(2048));
        assert_eq!(bar.length(), None);
    }

    #[test]
    fn fit_message_keeps_short_text_untouched() {
        assert_eq!(fit_message("short", 10), "short");
    }

    #[test]
    fn fit_message_marks_a_cut_with_an_ellipsis() {
        assert_eq!(fit_message("abcdefghij", 5), "abcd…");
    }

    #[test]
    fn fit_message_drops_everything_when_there_is_no_room() {
        assert_eq!(fit_message("abcdef", 0), "");
    }
}
