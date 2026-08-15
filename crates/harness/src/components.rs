use crate::theme;
use eframe::egui::{
    self, Button, Color32, FontId, Frame, Margin, Pos2, Rect, RichText, Sense, Stroke,
};

#[derive(Clone, Copy)]
pub enum MessageSurface {
    User,
    Assistant,
    Error,
}

pub fn card() -> Frame {
    Frame::new()
        .fill(theme::SURFACE)
        .stroke(Stroke::new(1.0, theme::BORDER))
        .corner_radius(12)
        .inner_margin(Margin::same(14))
}

fn message_text_color(surface: MessageSurface) -> Color32 {
    match surface {
        MessageSurface::User => theme::SIDEBAR,
        MessageSurface::Assistant => theme::TEXT,
        MessageSurface::Error => theme::ERROR,
    }
}

pub fn chat_bubble(
    ui: &mut egui::Ui,
    id: egui::Id,
    surface: MessageSurface,
    text: &str,
    align_right: bool,
) -> bool {
    let max_width = (ui.available_width() * 0.56).clamp(260.0, 560.0);
    let width = (text.chars().count() as f32 * 7.2 + 94.0).clamp(150.0, max_width);
    let text_color = message_text_color(surface);
    let galley = ui.fonts(|fonts| {
        fonts.layout(
            text.to_owned(),
            FontId::proportional(15.0),
            text_color,
            width - 48.0,
        )
    });
    let height = galley.size().y + 24.0;
    let (row, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), height), Sense::hover());
    let rect = bubble_rect(row, width, align_right);
    let (fill, stroke) = match surface {
        MessageSurface::User => (theme::BANANA, Stroke::NONE),
        MessageSurface::Assistant => (theme::SURFACE, Stroke::new(1.0, theme::BORDER)),
        MessageSurface::Error => (
            Color32::from_rgb(66, 38, 32),
            Stroke::new(1.0, theme::ERROR),
        ),
    };
    ui.painter()
        .rect(rect, 18.0, fill, stroke, egui::StrokeKind::Inside);
    ui.painter()
        .galley(rect.min + egui::vec2(14.0, 12.0), galley, text_color);
    let close = Rect::from_min_size(
        egui::pos2(rect.right() - 26.0, rect.top() + 5.0),
        egui::vec2(20.0, 20.0),
    );
    let response = ui.interact(close, id.with("delete"), Sense::click());
    let delete_color = if response.hovered() {
        theme::ERROR
    } else {
        theme::MUTED
    };
    ui.painter().text(
        close.center(),
        egui::Align2::CENTER_CENTER,
        "×",
        FontId::proportional(17.0),
        delete_color,
    );
    response.on_hover_text("Remove message").clicked()
}

fn bubble_rect(row: Rect, width: f32, align_right: bool) -> Rect {
    let left = if align_right {
        row.right() - width
    } else {
        row.left()
    };
    Rect::from_min_size(Pos2::new(left, row.top()), egui::vec2(width, row.height()))
}

pub fn primary(label: impl Into<String>) -> Button<'static> {
    Button::new(RichText::new(label).color(theme::SIDEBAR).strong())
        .fill(theme::BANANA)
        .stroke(Stroke::NONE)
}

pub fn muted(text: impl Into<String>) -> RichText {
    RichText::new(text).color(theme::MUTED)
}

pub fn status(ui: &mut egui::Ui, online: bool, text: &str) {
    ui.horizontal(|ui| {
        ui.colored_label(if online { theme::SUCCESS } else { theme::MUTED }, "●");
        ui.label(muted(text));
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bubble_geometry_keeps_outgoing_messages_on_the_right() {
        let row = Rect::from_min_size(Pos2::new(20.0, 40.0), egui::vec2(800.0, 64.0));
        let incoming = bubble_rect(row, 260.0, false);
        let outgoing = bubble_rect(row, 260.0, true);
        assert_eq!(incoming.left(), row.left());
        assert_eq!(outgoing.right(), row.right());
        assert_eq!(incoming.width(), outgoing.width());
    }
}
