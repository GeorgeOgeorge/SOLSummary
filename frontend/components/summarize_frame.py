from customtkinter import CTkLabel, CTkButton, CTkFrame, filedialog

from backend.file_summarizer import ArticleSummarizer


class SummarizeFrame(CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        self.select_article_button = CTkButton(
            master=self,
            command=self.select_file_path,
            text="Choose Article",
            font=("Roboto", 14)
        )
        self.select_article_button.grid(row=0, column=0)

        self.chosen_article_label = CTkLabel(
            master=self,
            text="No article selected yet",
            font=("Roboto", 14),
            text_color="gray"
        )
        self.chosen_article_label.grid(row=1, column=0, sticky="w")

        self.select_output_button = CTkButton(
            master=self,
            command=self.select_output_path,
            text="Choose Output Location (PDF)",
            font=("Roboto", 14)
        )
        self.select_output_button.grid(row=2, column=0)

        self.chosen_output_label = CTkLabel(
            master=self,
            text="No output location selected",
            font=("Roboto", 14),
            text_color="gray"
        )
        self.chosen_output_label.grid(row=3, column=0, sticky="w")

        self.summarize_button = CTkButton(
            master=self,
            command=self.sum_article,
            text="Summarize",
            font=("Roboto", 16),
            fg_color=None,
        )
        self.summarize_button.grid(row=4, column=0, pady=20)

        self.file_path = None
        self.output_path = None
        self.article_summarizer = ArticleSummarizer()

    def select_file_path(self) -> None:
        """selects file path to be summarized"""
        file_path = filedialog.askopenfilename(
            initialdir="~/",
            title="Select a PDF Article",
            filetypes=(("PDF Articles", "*.pdf*"), ("all files", "*.*"))
        )
        self.file_path = file_path
        self.chosen_article_label.configure(text=file_path or "No article selected yet")

    def select_output_path(self) -> None:
        """selects directory path to dump file summarize output file"""
        file_path = filedialog.askdirectory(
            initialdir="~/",
            title="Select Output Directory",
            mustexist=True
        )
        self.output_path = file_path
        self.chosen_output_label.configure(text=file_path or "No output location selected")

    def sum_article(self):
        """calls summarizer"""
        if self.file_path and self.output_path:
            self.article_summarizer.summarize_file(file_path=self.file_path, output_path=self.output_path)
            self.chosen_output_label.configure(text="Text Summarized! (Check Output Directory)")
        else:
            print("Please select both an article and an output location.")
