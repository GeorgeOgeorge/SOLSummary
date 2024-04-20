import customtkinter
from customtkinter import CTkFrame

from frontend.components.navigation import Navigation
from frontend.components.search_frame import SearchFrame
from frontend.components.summarize_frame import SummarizeFrame

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("dark-blue")


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.title("Auto Resumo")
        self.geometry("700x450")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        self.navigation_frame = Navigation(master=self, corner_radius=5)
        self.navigation_frame.grid(row=0, column=0, sticky="nsew")
        self.navigation_frame.grid_rowconfigure(4, weight=1)

        self.search_frame = SearchFrame(master=self)
        self.summarize_frame = SummarizeFrame(master=self)

        self.render_frame(frame_name="summarize_frame")

    def render_frame(self, frame_name: str) -> None:
        ignore_widgets = [self.navigation_frame]
        self.navigation_frame.search_frame_button.configure(
            fg_color=("gray75", "gray25") if frame_name == "search_frame" else "transparent"
        )
        self.navigation_frame.summarize_frame_button.configure(
            fg_color=("gray75", "gray25") if frame_name == "summarize_frame" else "transparent"
        )

        try:
            frame = self.__getattribute__(frame_name)
        except Exception:
            print(f"{frame_name} unlisted! unrendering all frames!")
        else:
            frame.grid(row=0, column=1, sticky="nsew")
            ignore_widgets.append(frame)

        for widget in self.winfo_children():
            if widget not in ignore_widgets and issubclass(type(widget), CTkFrame):
                widget.grid_forget()
