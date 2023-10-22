from customtkinter import CTkLabel, CTkButton, CTkFrame, filedialog

from summarizer import ArticleSummarizer


class SummarizeFrame(CTkFrame):
    file_path = None
    output_path = None
    article_summarizer = ArticleSummarizer()

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        self.select_article_button = CTkButton(
            master=self,
            command=self.select_file_path,
            text="escolha artigo",
            font=("roboto", 13.5)
        )
        self.select_article_button.grid(row=0, column=0)

        self.chosen_input_label = CTkLabel(
            master=self,
            text="Escolha artigo a ser resumido",
            font=("roboto", 13.5)
        )
        self.chosen_input_label.grid(row=1, column=0)

        self.select_output_button = CTkButton(
            master=self,
            command=self.select_output_path,
            text="escolha local de saide de pdf",
            font=("roboto", 13.5)
        )
        self.select_output_button.grid(row=2, column=0)

        self.chosen_output_label = CTkLabel(
            master=self,
            text="escolha local se salvamento do resumo",
            font=("roboto", 13.5)
        )
        self.chosen_output_label.grid(row=3, column=0)

        self.button = CTkButton(
            master=self,
            command=self.sum_article,
            text="Resumir",
            font=("roboto", 15)
        )
        self.button.grid(row=4, column=0)

    def select_file_path(self) -> None:
        """selects file path to be summarized"""
        file_path = filedialog.askopenfilename(
            initialdir="~/",
            title="Select a article",
            filetypes=(("Articles", "*.pdf*"), ("all files", "*.*"))
        )
        self.file_path = file_path
        self.chosen_input_label.configure(text=file_path)

    def select_output_path(self) -> None:
        """selects directory path to dump file summarize output file"""
        file_path = filedialog.askdirectory(
            initialdir="~/",
            title="Select a article",
            mustexist=True
        )
        self.output_path = file_path
        self.chosen_output_label.configure(text=file_path)

    def sum_article(self):
        """calls summarizer"""
        self.article_summarizer.summarize_file(file_path=self.file_path, output_path=self.output_path)
        self.chosen_output_label.configure(text="texto resumido")
