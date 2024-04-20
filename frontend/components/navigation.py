from customtkinter import CTkLabel, CTkFont, CTkButton, CTkFrame


class Navigation(CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        self.navigation_frame_label = CTkLabel(
            self,
            text="AutoResumo",
            compound="left",
            font=CTkFont(size=15, weight="bold")
        )
        self.navigation_frame_label.grid(row=0, column=0, padx=20, pady=20)

        self.search_frame_button = CTkButton(
            self,
            corner_radius=0,
            height=40,
            border_spacing=10,
            text="Pesquiar Artigo",
            fg_color="transparent",
            text_color=("gray10", "gray90"),
            hover_color=("gray70", "gray30"),
            anchor="w",
            command=lambda: self._render_frame("search_frame")
        )
        self.search_frame_button.grid(row=1, column=0, sticky="ew")

        self.summarize_frame_button = CTkButton(
            self,
            corner_radius=0,
            height=40,
            border_spacing=10,
            text="Resumir Artigo",
            fg_color=("gray75", "gray25"),
            text_color=("gray10", "gray90"),
            hover_color=("gray70", "gray30"),
            anchor="w",
            command=lambda: self._render_frame("summarize_frame")
        )
        self.summarize_frame_button.grid(row=2, column=0, sticky="ew")

    def _render_frame(self, frame_name: str) -> None:
        """Renders Frame to be shown

        Args:
            frame_name (str): frame name to be show
        """
        self.master.render_frame(frame_name)
