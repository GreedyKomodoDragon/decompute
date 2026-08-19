#![cfg_attr(not(target_os = "macos"), allow(dead_code))]

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("Decompute Harness is supported on macOS only.");
}

#[cfg(target_os = "macos")]
mod components;
#[cfg(target_os = "macos")]
mod theme;

#[cfg(target_os = "macos")]
mod app {
    use super::{components, theme};
    use eframe::egui;
    use futures_util::StreamExt;
    use serde::{Deserialize, Serialize};
    use std::{sync::mpsc, time::Duration};
    use tokio_util::sync::CancellationToken;
    use uuid::Uuid;

    const STORAGE_KEY: &str = "decompute_harness";

    #[derive(Clone, Serialize, Deserialize)]
    struct Settings {
        endpoint: String,
        model: String,
        context_budget: usize,
        max_tokens: usize,
    }

    impl Default for Settings {
        fn default() -> Self {
            Self {
                endpoint: "http://127.0.0.1:8000".into(),
                model: "qwen2.5-0.5b-instruct-q4-k-m".into(),
                context_budget: 2_048,
                max_tokens: 256,
            }
        }
    }

    #[derive(Clone, Serialize, Deserialize)]
    struct Conversation {
        id: Uuid,
        title: String,
        system_enabled: bool,
        system_harness: String,
        messages: Vec<Message>,
    }

    impl Conversation {
        fn new() -> Self {
            Self {
                id: Uuid::new_v4(),
                title: "New conversation".into(),
                system_enabled: false,
                system_harness: String::new(),
                messages: vec![],
            }
        }
    }

    #[derive(Clone, Serialize, Deserialize)]
    struct Message {
        id: Uuid,
        role: Role,
        content: String,
    }

    #[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    enum Role {
        User,
        Assistant,
        Error,
    }

    #[derive(Serialize, Deserialize)]
    struct SavedState {
        settings: Settings,
        conversations: Vec<Conversation>,
        selected: Option<Uuid>,
    }

    impl Default for SavedState {
        fn default() -> Self {
            let conversation = Conversation::new();
            Self {
                settings: Settings::default(),
                selected: Some(conversation.id),
                conversations: vec![conversation],
            }
        }
    }

    enum Event {
        Models(Result<Vec<String>, String>),
        Delta {
            conversation: Uuid,
            message: Uuid,
            text: String,
        },
        Completed {
            conversation: Uuid,
        },
        Failed {
            conversation: Uuid,
            message: String,
        },
    }

    pub struct HarnessApp {
        state: SavedState,
        draft: String,
        models: Vec<String>,
        status: String,
        connected: bool,
        events: mpsc::Receiver<Event>,
        event_tx: mpsc::Sender<Event>,
        cancel: Option<CancellationToken>,
        streaming_conversation: Option<Uuid>,
        show_settings: bool,
    }

    impl HarnessApp {
        pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
            theme::apply(&cc.egui_ctx);
            let state = cc
                .storage
                .and_then(|s| eframe::get_value(s, STORAGE_KEY))
                .unwrap_or_default();
            let (event_tx, events) = mpsc::channel();
            let mut app = Self {
                state,
                draft: String::new(),
                models: vec![],
                status: "Not connected".into(),
                connected: false,
                events,
                event_tx,
                cancel: None,
                streaming_conversation: None,
                show_settings: false,
            };
            app.refresh_models();
            app
        }

        fn selected_mut(&mut self) -> Option<&mut Conversation> {
            let id = self.state.selected?;
            self.state
                .conversations
                .iter_mut()
                .find(|conversation| conversation.id == id)
        }
        fn selected(&self) -> Option<&Conversation> {
            let id = self.state.selected?;
            self.state
                .conversations
                .iter()
                .find(|conversation| conversation.id == id)
        }
        fn new_conversation(&mut self) {
            let conversation = Conversation::new();
            self.state.selected = Some(conversation.id);
            self.state.conversations.push(conversation);
        }
        fn refresh_models(&mut self) {
            self.status = "Loading models…".into();
            let endpoint = self.state.settings.endpoint.clone();
            let tx = self.event_tx.clone();
            spawn(async move {
                let _ = tx.send(Event::Models(fetch_models(&endpoint).await));
            });
        }
        fn send(&mut self) {
            if self.draft.trim().is_empty() || self.cancel.is_some() {
                return;
            }
            let settings = self.state.settings.clone();
            let content = std::mem::take(&mut self.draft);
            let Some(conversation) = self.selected_mut() else {
                return;
            };
            let user = Message {
                id: Uuid::new_v4(),
                role: Role::User,
                content,
            };
            if conversation.messages.is_empty() {
                conversation.title = user.content.chars().take(36).collect();
            }
            conversation.messages.push(user);
            let assistant = Uuid::new_v4();
            conversation.messages.push(Message {
                id: assistant,
                role: Role::Assistant,
                content: String::new(),
            });
            let request = Request::from_conversation(&settings, conversation);
            let id = conversation.id;
            let cancel = CancellationToken::new();
            self.cancel = Some(cancel.clone());
            self.streaming_conversation = Some(id);
            self.status = "Generating…".into();
            let tx = self.event_tx.clone();
            spawn(async move {
                stream_chat(settings.endpoint, request, id, assistant, cancel, tx).await;
            });
        }
        fn process_events(&mut self) {
            while let Ok(event) = self.events.try_recv() {
                match event {
                    Event::Models(Ok(models)) => {
                        self.models = models;
                        self.status = "Connected".into();
                        self.connected = true;
                        self.state.settings.model =
                            selected_discovered_model(&self.state.settings.model, &self.models);
                    }
                    Event::Models(Err(error)) => {
                        self.status = format!("Model discovery failed: {error}");
                        self.connected = false;
                    }
                    Event::Delta {
                        conversation,
                        message,
                        text,
                    } => {
                        if let Some(c) = self
                            .state
                            .conversations
                            .iter_mut()
                            .find(|c| c.id == conversation)
                            && let Some(m) = c.messages.iter_mut().find(|m| m.id == message)
                        {
                            m.content.push_str(&text);
                        }
                    }
                    Event::Completed { conversation } => {
                        if self.streaming_conversation == Some(conversation) {
                            self.cancel = None;
                            self.streaming_conversation = None;
                            self.status = "Completed".into();
                        }
                    }
                    Event::Failed {
                        conversation,
                        message,
                    } => {
                        if self.streaming_conversation == Some(conversation) {
                            self.cancel = None;
                            self.streaming_conversation = None;
                            self.status = "Request failed".into();
                        }
                        if let Some(c) = self
                            .state
                            .conversations
                            .iter_mut()
                            .find(|c| c.id == conversation)
                        {
                            c.messages.push(Message {
                                id: Uuid::new_v4(),
                                role: Role::Error,
                                content: message,
                            });
                        }
                    }
                }
            }
        }
    }

    impl eframe::App for HarnessApp {
        fn save(&mut self, storage: &mut dyn eframe::Storage) {
            eframe::set_value(storage, STORAGE_KEY, &self.state);
        }
        fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
            self.process_events();
            ctx.request_repaint_after(Duration::from_millis(50));
            let sidebar_width = if ctx.available_rect().width() < 860.0 {
                208.0
            } else {
                248.0
            };
            egui::SidePanel::left("conversations")
                .exact_width(sidebar_width)
                .frame(
                    egui::Frame::new()
                        .fill(theme::SIDEBAR)
                        .stroke(egui::Stroke::new(1.0, theme::BORDER))
                        .inner_margin(egui::Margin::symmetric(14, theme::SPACE_16 as i8)),
                )
                .show(ctx, |ui| {
                    egui::TopBottomPanel::bottom("sidebar_footer")
                        .frame(egui::Frame::NONE)
                        .show_inside(ui, |ui| {
                            ui.separator();
                            ui.add_space(theme::SPACE_8);
                            components::status(ui, self.connected, &self.status);
                            ui.add_space(theme::SPACE_8);
                            if ui
                                .add_sized(
                                    [ui.available_width(), 36.0],
                                    components::secondary("Settings"),
                                )
                                .clicked()
                            {
                                self.show_settings = true;
                            }
                        });

                    ui.horizontal(|ui| {
                        ui.heading("decompute");
                        ui.add_space(theme::SPACE_4);
                        ui.label(components::muted("HARNESS"));
                    });
                    ui.add_space(theme::SPACE_16);
                    if ui
                        .add_sized(
                            [ui.available_width(), 42.0],
                            components::primary("+  New chat"),
                        )
                        .clicked()
                    {
                        self.new_conversation();
                    }
                    ui.add_space(theme::SPACE_16);
                    ui.label(components::muted("RECENT"));
                    ui.add_space(theme::SPACE_8);
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        for conversation in &self.state.conversations {
                            let title = navigation_title(&conversation.title, sidebar_width);
                            if components::navigation_item(
                                ui,
                                &title,
                                self.state.selected == Some(conversation.id),
                            )
                            .clicked()
                            {
                                self.state.selected = Some(conversation.id);
                            }
                            ui.add_space(theme::SPACE_4);
                        }
                    });
                });
            egui::CentralPanel::default().show(ctx, |ui| {
                let Some(conversation) = self.selected() else {
                    return;
                };
                let title = conversation.title.clone();
                let chars = conversation
                    .messages
                    .iter()
                    .map(|message| message.content.len())
                    .sum::<usize>()
                    + if conversation.system_enabled {
                        conversation.system_harness.len()
                    } else {
                        0
                    };
                let estimate = approximate_tokens(chars);

                conversation_column(ui, |ui| {
                    ui.add_space(theme::SPACE_8);
                    ui.horizontal(|ui| {
                        ui.heading(title);
                        ui.add_space(theme::SPACE_8);
                        ui.label(components::muted(format!(
                            "~{estimate} / {} tokens",
                            self.state.settings.context_budget
                        )));
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            egui::ComboBox::from_id_salt("model")
                                .width(176.0)
                                .selected_text(&self.state.settings.model)
                                .show_ui(ui, |ui| {
                                    for model in &self.models {
                                        ui.selectable_value(
                                            &mut self.state.settings.model,
                                            model.clone(),
                                            model,
                                        );
                                    }
                                });
                        });
                    });
                    ui.add_space(theme::SPACE_12);
                    ui.separator();
                    ui.add_space(theme::SPACE_8);
                });

                let mut send = false;
                let mut clear = false;
                egui::TopBottomPanel::bottom("composer")
                    .show_separator_line(false)
                    .show_inside(ui, |ui| {
                    conversation_column(ui, |ui| {
                        let input_id = ui.make_persistent_id("composer_input");
                        let focused = ui.memory(|memory| memory.has_focus(input_id));
                        components::composer_card(focused).show(ui, |ui| {
                            let rows = self.draft.lines().count().clamp(1, 5);
                            let response = ui.add(
                                egui::TextEdit::multiline(&mut self.draft)
                                    .id(input_id)
                                    .desired_width(f32::INFINITY)
                                    .desired_rows(rows)
                                    .hint_text("Message Decompute…"),
                            );
                            send |= response.has_focus()
                                && ui.input(|input| {
                                    input.key_pressed(egui::Key::Enter) && !input.modifiers.shift
                                });
                            ui.add_space(theme::SPACE_8);
                            ui.horizontal(|ui| {
                                ui.label(components::muted(
                                    "Local inference · no hidden system prompt",
                                ));
                                ui.add_space(theme::SPACE_8);
                                ui.label(components::muted(
                                    "Enter to send · Shift+Enter for a new line",
                                ));
                                ui.with_layout(
                                    egui::Layout::right_to_left(egui::Align::Center),
                                    |ui| {
                                        send |= ui
                                            .add_enabled(
                                                self.cancel.is_none()
                                                    && !self.draft.trim().is_empty(),
                                                components::primary("Send"),
                                            )
                                            .clicked();
                                        if ui
                                            .add_enabled(
                                                self.cancel.is_some(),
                                                components::secondary("Stop"),
                                            )
                                            .on_hover_text("Stop the active generation")
                                            .clicked()
                                            && let Some(cancel) = &self.cancel
                                        {
                                            cancel.cancel();
                                        }
                                        if ui.add(components::secondary("Clear")).clicked() {
                                            clear = true;
                                        }
                                    },
                                );
                            });
                        });
                    });
                    ui.add_space(theme::SPACE_12);
                    });

                if estimate + self.state.settings.max_tokens > self.state.settings.context_budget {
                    conversation_column(ui, |ui| {
                        ui.colored_label(
                            egui::Color32::YELLOW,
                            "This request may exceed the selected context budget. Remove messages or increase the budget.",
                        );
                    });
                }
                let mut remove_message = None;
                egui::ScrollArea::vertical()
                    .auto_shrink([false; 2])
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        conversation_column(ui, |ui| {
                            if let Some(conversation) = self.selected() {
                                ui.add_space(theme::SPACE_24);
                                if conversation.messages.is_empty() {
                                    ui.add_space(theme::SPACE_32 * 2.0);
                                    ui.heading("Start a conversation");
                                    ui.label(components::muted(
                                        "Decompute has no default system prompt. Ask anything to begin.",
                                    ));
                                }
                                for message in &conversation.messages {
                                    let live_placeholder = message.role == Role::Assistant
                                        && message.content.is_empty()
                                        && self.streaming_conversation == Some(conversation.id);
                                    if message.role == Role::Assistant
                                        && message.content.is_empty()
                                        && !live_placeholder
                                    {
                                        continue;
                                    }
                                    let (surface, align_right) = match message.role {
                                        Role::User => (components::MessageSurface::User, true),
                                        Role::Assistant => {
                                            (components::MessageSurface::Assistant, false)
                                        }
                                        Role::Error => (components::MessageSurface::Error, false),
                                    };
                                    let text = if live_placeholder {
                                        "Thinking…"
                                    } else {
                                        &message.content
                                    };
                                    let response = components::chat_bubble(ui, surface, text, align_right);
                                    response.context_menu(|ui| {
                                        if ui.button("Delete message").clicked() {
                                            remove_message = Some(message.id);
                                            ui.close_menu();
                                        }
                                    });
                                    ui.add_space(theme::SPACE_12);
                                }
                            }
                        });
                    });
                if let Some(id) = remove_message
                    && let Some(conversation) = self.selected_mut()
                {
                    conversation.messages.retain(|message| message.id != id);
                }
                if clear && let Some(conversation) = self.selected_mut() {
                    conversation.messages.clear();
                }
                if send && self.cancel.is_none() && !self.draft.trim().is_empty() {
                    self.send();
                }
            });
            if self.show_settings {
                let mut open = self.show_settings;
                let mut refresh = false;
                egui::Window::new("Connection settings")
                    .open(&mut open)
                    .resizable(false)
                    .show(ctx, |ui| {
                        ui.label(components::muted("OPENAI-COMPATIBLE ENDPOINT"));
                        ui.text_edit_singleline(&mut self.state.settings.endpoint);
                        if ui.button("Connect / refresh models").clicked() {
                            refresh = true;
                        }
                        ui.separator();
                        ui.add(
                            egui::DragValue::new(&mut self.state.settings.context_budget)
                                .range(128..=131_072)
                                .prefix("Context budget: "),
                        );
                        ui.add(
                            egui::DragValue::new(&mut self.state.settings.max_tokens)
                                .range(1..=8_192)
                                .prefix("Max output: "),
                        );
                        ui.separator();
                        ui.collapsing("System harness", |ui| {
                            if let Some(conversation) = self.selected_mut() {
                                ui.checkbox(
                                    &mut conversation.system_enabled,
                                    "Include a system message for this chat",
                                );
                                ui.add_enabled(
                                    conversation.system_enabled,
                                    egui::TextEdit::multiline(&mut conversation.system_harness)
                                        .desired_rows(4)
                                        .hint_text("No hidden default prompt."),
                                );
                            }
                        });
                        ui.add_space(4.0);
                        ui.label(components::muted("Settings and chats stay on this Mac."));
                    });
                self.show_settings = open;
                if refresh {
                    self.refresh_models();
                }
            }
        }
    }

    const CONTENT_MAX_WIDTH: f32 = 980.0;

    fn conversation_column(ui: &mut egui::Ui, add_contents: impl FnOnce(&mut egui::Ui)) {
        let width = ui.available_width().min(CONTENT_MAX_WIDTH);
        let outer_width = ui.available_width();
        ui.horizontal(|ui| {
            ui.add_space(((outer_width - width) / 2.0).max(0.0));
            ui.allocate_ui_with_layout(
                egui::vec2(width, 0.0),
                egui::Layout::top_down(egui::Align::LEFT),
                add_contents,
            );
        });
    }

    fn navigation_title(title: &str, sidebar_width: f32) -> String {
        let max_chars = if sidebar_width < 240.0 { 20 } else { 27 };
        let mut title = title.trim().chars();
        let Some(first) = title.next() else {
            return "New conversation".into();
        };
        let normalized = format!("{}{}", first.to_uppercase(), title.collect::<String>());
        if normalized.chars().count() > max_chars {
            format!(
                "{}…",
                normalized.chars().take(max_chars - 1).collect::<String>()
            )
        } else {
            normalized
        }
    }

    #[derive(Serialize)]
    struct Request {
        model: String,
        messages: Vec<ApiMessage>,
        max_tokens: usize,
        stream: bool,
    }
    #[derive(Serialize)]
    struct ApiMessage {
        role: &'static str,
        content: String,
    }
    impl Request {
        fn from_conversation(settings: &Settings, conversation: &Conversation) -> Self {
            let mut messages = vec![];
            if conversation.system_enabled && !conversation.system_harness.trim().is_empty() {
                messages.push(ApiMessage {
                    role: "system",
                    content: conversation.system_harness.clone(),
                });
            }
            messages.extend(conversation.messages.iter().filter_map(
                |message| match message.role {
                    Role::User => Some(ApiMessage {
                        role: "user",
                        content: message.content.clone(),
                    }),
                    Role::Assistant if !message.content.is_empty() => Some(ApiMessage {
                        role: "assistant",
                        content: message.content.clone(),
                    }),
                    _ => None,
                },
            ));
            Self {
                model: settings.model.clone(),
                messages,
                max_tokens: settings.max_tokens,
                stream: true,
            }
        }
    }

    fn approximate_tokens(characters: usize) -> usize {
        characters.div_ceil(4)
    }

    fn selected_discovered_model(current: &str, models: &[String]) -> String {
        if models.iter().any(|model| model == current) {
            current.to_owned()
        } else {
            models.first().cloned().unwrap_or_default()
        }
    }
    #[derive(Deserialize)]
    struct ModelList {
        data: Vec<Model>,
    }
    #[derive(Deserialize)]
    struct Model {
        id: String,
    }
    async fn fetch_models(endpoint: &str) -> Result<Vec<String>, String> {
        let response = reqwest::Client::new()
            .get(format!("{}/v1/models", endpoint.trim_end_matches('/')))
            .send()
            .await
            .map_err(|e| e.to_string())?
            .error_for_status()
            .map_err(|e| e.to_string())?
            .json::<ModelList>()
            .await
            .map_err(|e| e.to_string())?;
        Ok(response.data.into_iter().map(|m| m.id).collect())
    }
    async fn stream_chat(
        endpoint: String,
        request: Request,
        conversation: Uuid,
        message: Uuid,
        cancel: CancellationToken,
        tx: mpsc::Sender<Event>,
    ) {
        let client = reqwest::Client::new();
        let response = tokio::select! {
            _ = cancel.cancelled() => { let _ = tx.send(Event::Completed { conversation }); return; }
            response = client.post(format!("{}/v1/chat/completions", endpoint.trim_end_matches('/'))).json(&request).send() => response,
        };
        let response = match response {
            Ok(response) if response.status().is_success() => response,
            Ok(response) => {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                let _ = tx.send(Event::Failed {
                    conversation,
                    message: format!("Server returned {status}: {body}"),
                });
                return;
            }
            Err(error) => {
                let _ = tx.send(Event::Failed {
                    conversation,
                    message: format!("Request failed: {error}"),
                });
                return;
            }
        };
        let mut stream = response.bytes_stream();
        let mut buffer = Vec::new();
        loop {
            let next = tokio::select! {
                _ = cancel.cancelled() => { let _ = tx.send(Event::Completed { conversation }); return; }
                next = stream.next() => next,
            };
            let Some(next) = next else {
                let _ = tx.send(Event::Failed {
                    conversation,
                    message: "Stream ended before [DONE]".into(),
                });
                return;
            };
            let bytes = match next {
                Ok(bytes) => bytes,
                Err(error) => {
                    let _ = tx.send(Event::Failed {
                        conversation,
                        message: format!("Stream read failed: {error}"),
                    });
                    return;
                }
            };
            buffer.extend_from_slice(&bytes);
            while let Some(end) = buffer.windows(2).position(|window| window == b"\n\n") {
                let frame = buffer.drain(..end + 2).collect::<Vec<_>>();
                let frame = String::from_utf8_lossy(&frame);
                let data = frame
                    .lines()
                    .filter_map(|line| line.strip_prefix("data:"))
                    .map(str::trim_start)
                    .collect::<Vec<_>>()
                    .join("\n");
                if data.is_empty() {
                    continue;
                }
                if data == "[DONE]" {
                    let _ = tx.send(Event::Completed { conversation });
                    return;
                }
                let value: serde_json::Value = match serde_json::from_str(&data) {
                    Ok(value) => value,
                    Err(error) => {
                        let _ = tx.send(Event::Failed {
                            conversation,
                            message: format!("Invalid SSE JSON: {error}"),
                        });
                        return;
                    }
                };
                if let Some(text) = value
                    .pointer("/choices/0/delta/content")
                    .and_then(serde_json::Value::as_str)
                {
                    let _ = tx.send(Event::Delta {
                        conversation,
                        message,
                        text: text.to_owned(),
                    });
                }
            }
        }
    }
    fn spawn(future: impl std::future::Future<Output = ()> + Send + 'static) {
        std::thread::spawn(move || {
            tokio::runtime::Runtime::new()
                .expect("create network runtime")
                .block_on(future);
        });
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn request_has_no_system_message_by_default() {
            let mut conversation = Conversation::new();
            conversation.messages.push(Message {
                id: Uuid::nil(),
                role: Role::User,
                content: "hello".into(),
            });
            let request = Request::from_conversation(&Settings::default(), &conversation);
            assert_eq!(request.messages.len(), 1);
            assert_eq!(request.messages[0].role, "user");
        }

        #[test]
        fn enabled_system_harness_is_visible_leading_message() {
            let mut conversation = Conversation::new();
            conversation.system_enabled = true;
            conversation.system_harness = "Answer in haiku.".into();
            conversation.messages.push(Message {
                id: Uuid::nil(),
                role: Role::User,
                content: "hello".into(),
            });
            let request = Request::from_conversation(&Settings::default(), &conversation);
            assert_eq!(request.messages[0].role, "system");
            assert_eq!(request.messages[0].content, "Answer in haiku.");
        }

        #[test]
        fn token_estimate_rounds_up() {
            assert_eq!(approximate_tokens(0), 0);
            assert_eq!(approximate_tokens(1), 1);
            assert_eq!(approximate_tokens(4), 1);
            assert_eq!(approximate_tokens(5), 2);
        }

        #[test]
        fn discovery_replaces_a_stale_model_and_preserves_an_available_one() {
            let models = vec![
                "qwen2.5-0.5b-instruct-q4-k-m".into(),
                "qwen2.5-1.5b-instruct-q4-k-m".into(),
            ];
            assert_eq!(
                selected_discovered_model("tiny-model", &models),
                "qwen2.5-0.5b-instruct-q4-k-m"
            );
            assert_eq!(
                selected_discovered_model("qwen2.5-1.5b-instruct-q4-k-m", &models),
                "qwen2.5-1.5b-instruct-q4-k-m"
            );
        }

        #[test]
        fn conversation_column_is_centered_and_never_exceeds_its_maximum_width() {
            let context = egui::Context::default();
            let mut rects = None;
            let _ = context.run(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(1_400.0, 700.0),
                    )),
                    ..Default::default()
                },
                |context| {
                    egui::CentralPanel::default().show(context, |ui| {
                        let panel = ui.max_rect();
                        conversation_column(ui, |ui| {
                            let (rect, _) = ui.allocate_exact_size(
                                egui::vec2(ui.available_width(), 10.0),
                                egui::Sense::hover(),
                            );
                            rects = Some((panel, rect));
                        });
                    });
                },
            );

            let (panel, column) = rects.expect("the test column renders");
            assert!(column.width() <= CONTENT_MAX_WIDTH + 1.0);
            assert!(
                (column.center().x - panel.center().x).abs() <= 1.0,
                "conversation column must remain centred in its usable panel"
            );
        }
    }
}

#[cfg(target_os = "macos")]
fn main() -> eframe::Result<()> {
    eframe::run_native(
        "Decompute Harness",
        eframe::NativeOptions::default(),
        Box::new(|cc| Ok(Box::new(app::HarnessApp::new(cc)))),
    )
}
