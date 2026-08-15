use eframe::egui::{self, Color32, FontId, TextStyle, Vec2};

pub const CANVAS: Color32 = Color32::from_rgb(24, 24, 20);
pub const SIDEBAR: Color32 = Color32::from_rgb(18, 18, 15);
pub const SURFACE: Color32 = Color32::from_rgb(36, 36, 30);
pub const SURFACE_RAISED: Color32 = Color32::from_rgb(45, 44, 36);
pub const BORDER: Color32 = Color32::from_rgb(74, 72, 57);
pub const TEXT: Color32 = Color32::from_rgb(246, 243, 226);
pub const MUTED: Color32 = Color32::from_rgb(169, 165, 143);
pub const BANANA: Color32 = Color32::from_rgb(246, 211, 70);
pub const BANANA_HOVER: Color32 = Color32::from_rgb(255, 225, 105);
pub const ERROR: Color32 = Color32::from_rgb(244, 125, 101);
pub const SUCCESS: Color32 = Color32::from_rgb(143, 204, 139);

pub fn apply(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();
    style.spacing.item_spacing = Vec2::new(10.0, 10.0);
    style.spacing.button_padding = Vec2::new(12.0, 8.0);
    style.spacing.interact_size.y = 34.0;
    style
        .text_styles
        .insert(TextStyle::Body, FontId::proportional(15.0));
    style
        .text_styles
        .insert(TextStyle::Button, FontId::proportional(14.0));
    style
        .text_styles
        .insert(TextStyle::Heading, FontId::proportional(22.0));
    let mut visuals = egui::Visuals::dark();
    visuals.panel_fill = CANVAS;
    visuals.window_fill = SURFACE;
    visuals.faint_bg_color = SURFACE;
    visuals.extreme_bg_color = SIDEBAR;
    visuals.override_text_color = Some(TEXT);
    visuals.selection.bg_fill = BANANA.gamma_multiply(0.42);
    visuals.selection.stroke.color = BANANA;
    visuals.widgets.noninteractive.bg_fill = CANVAS;
    visuals.widgets.noninteractive.fg_stroke.color = TEXT;
    visuals.widgets.inactive.bg_fill = SURFACE;
    visuals.widgets.inactive.fg_stroke.color = TEXT;
    visuals.widgets.inactive.bg_stroke.color = BORDER;
    visuals.widgets.hovered.bg_fill = SURFACE_RAISED;
    visuals.widgets.hovered.bg_stroke.color = BANANA_HOVER;
    visuals.widgets.active.bg_fill = BANANA.gamma_multiply(0.45);
    visuals.widgets.active.bg_stroke.color = BANANA;
    style.visuals = visuals;
    ctx.set_style(style);
}
