use eframe::egui::{self, Color32, FontId, Stroke, TextStyle, Vec2};

pub const CANVAS: Color32 = Color32::from_rgb(22, 22, 19);
pub const SIDEBAR: Color32 = Color32::from_rgb(18, 18, 16);
pub const SURFACE: Color32 = Color32::from_rgb(31, 31, 27);
pub const SURFACE_RAISED: Color32 = Color32::from_rgb(42, 42, 36);
pub const BORDER: Color32 = Color32::from_rgb(77, 76, 68);
pub const TEXT: Color32 = Color32::from_rgb(244, 242, 229);
pub const MUTED: Color32 = Color32::from_rgb(177, 173, 150);
pub const BANANA: Color32 = Color32::from_rgb(248, 211, 68);
pub const BANANA_HOVER: Color32 = Color32::from_rgb(255, 226, 108);
pub const ERROR: Color32 = Color32::from_rgb(244, 125, 101);
pub const SUCCESS: Color32 = Color32::from_rgb(143, 204, 139);

pub const SPACE_4: f32 = 4.0;
pub const SPACE_8: f32 = 8.0;
pub const SPACE_12: f32 = 12.0;
pub const SPACE_16: f32 = 16.0;
pub const SPACE_24: f32 = 24.0;
pub const SPACE_32: f32 = 32.0;
pub const RADIUS_SMALL: u8 = 8;
pub const RADIUS_MEDIUM: u8 = 10;
pub const RADIUS_LARGE: u8 = 12;

pub fn apply(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();
    style.spacing.item_spacing = Vec2::new(SPACE_8, SPACE_8);
    style.spacing.button_padding = Vec2::new(SPACE_12, SPACE_8);
    style.spacing.interact_size.y = 36.0;
    style
        .text_styles
        .insert(TextStyle::Body, FontId::proportional(15.5));
    style
        .text_styles
        .insert(TextStyle::Button, FontId::proportional(14.5));
    style
        .text_styles
        .insert(TextStyle::Heading, FontId::proportional(21.0));
    let mut visuals = egui::Visuals::dark();
    visuals.panel_fill = CANVAS;
    visuals.window_fill = SURFACE;
    visuals.faint_bg_color = SURFACE;
    visuals.extreme_bg_color = SURFACE_RAISED;
    visuals.override_text_color = Some(TEXT);
    visuals.selection.bg_fill = BANANA.gamma_multiply(0.42);
    visuals.selection.stroke.color = BANANA;
    visuals.widgets.noninteractive.bg_fill = CANVAS;
    visuals.widgets.noninteractive.fg_stroke.color = TEXT;
    visuals.widgets.inactive.bg_fill = SURFACE;
    visuals.widgets.inactive.fg_stroke.color = TEXT;
    visuals.widgets.inactive.bg_stroke = Stroke::new(1.0, BORDER);
    visuals.widgets.hovered.bg_fill = SURFACE_RAISED;
    visuals.widgets.hovered.bg_stroke = Stroke::new(1.0, BANANA_HOVER.gamma_multiply(0.7));
    visuals.widgets.active.bg_fill = BANANA.gamma_multiply(0.45);
    visuals.widgets.active.bg_stroke = Stroke::new(1.0, BANANA);
    visuals.widgets.open.bg_fill = SURFACE_RAISED;
    visuals.widgets.open.bg_stroke = Stroke::new(1.0, BANANA.gamma_multiply(0.75));
    visuals.widgets.noninteractive.bg_stroke = Stroke::new(1.0, BORDER);
    visuals.window_stroke = Stroke::new(1.0, BORDER);
    visuals.window_corner_radius = RADIUS_MEDIUM.into();
    style.visuals = visuals;
    ctx.set_style(style);
}
