import tkinter as tk
import customtkinter as ctk
import requests
from bs4 import BeautifulSoup

search_url = "https://sol.sbc.org.br/busca/index.php/integrada/results?isAdvanced=1&archiveIds%5B%5D=1&query=&field-3=computador&field-15=&field-4=&field-14=&field-16="


class SearchFrame(ctk.CTkFrame):
    title_checkbox: ctk.CTkCheckBox
    title_input: ctk.CTkEntry

    summary_checkbox: ctk.CTkCheckBox
    summary_input: ctk.CTkEntry

    author_checkbox: ctk.CTkCheckBox
    author_input: ctk.CTkEntry

    search_button: ctk.CTkButton

    result_list: tk.Listbox

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        self.title_checkbox = ctk.CTkCheckBox(self, command=self.toogle_title_state, text="Search by Title:")
        self.title_checkbox.grid(row=0, column=0, sticky=tk.W)
        self.title_input = ctk.CTkEntry(self, state=tk.DISABLED, width=200, height=25)
        self.title_input.grid(row=0, column=1, columnspan=2)

        self.summary_checkbox = ctk.CTkCheckBox(self, command=self.toogle_summary_state, text="Search by Summary:")
        self.summary_checkbox.grid(row=1, column=0, sticky=tk.W)
        self.summary_input = ctk.CTkEntry(self, state=tk.DISABLED, width=200, height=25)
        self.summary_input.grid(row=1, column=1, columnspan=2)

        self.author_checkbox = ctk.CTkCheckBox(self, command=self.toogle_author_state, text="Search by Author:")
        self.author_checkbox.grid(row=2, column=0, sticky=tk.W)
        self.author_input = ctk.CTkEntry(self, state=tk.DISABLED, width=200, height=25)
        self.author_input.grid(row=2, column=1, columnspan=2)

        self.search_button = ctk.CTkButton(self, text="Search", command=self.search_articles)
        self.search_button.grid(row=3, column=0, columnspan=4, pady=15)

        self.result_list = tk.Listbox(
            master=self, height=10, width=38, bg="grey", fg="yellow", activestyle='dotbox', font="Helvetica",
        )
        self.result_list.insert(1, "Nachos")
        self.result_list.insert(2, "Nachos")
        self.result_list.insert(3, "Nachos")
        self.result_list.grid(row=4, column=0, columnspan=4)

    def toogle_title_state(self) -> None:
        self.title_input.configure(state=tk.NORMAL if self.title_checkbox.get() == 1 else tk.DISABLED)

    def toogle_summary_state(self) -> None:
        self.summary_input.configure(state=tk.NORMAL if self.summary_checkbox.get() == 1 else tk.DISABLED)

    def toogle_author_state(self) -> None:
        self.author_input.configure(state=tk.NORMAL if self.author_checkbox.get() == 1 else tk.DISABLED)

    def search_articles(self):
        article_dict = self.get_article_titles(search_url)
        breakpoint()

        print(article_dict)

    def get_article_titles(self, url: str) -> dict:
        """
        This function takes a URL as input and returns a dictionary containing
        article titles and their corresponding URLs.

        Args:
            url: The URL of the search page.

        Returns:
            A dictionary containing article titles and their corresponding URLs.
        """
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        articles = soup.find_all('div', class_='archive_title')
        article_data = {}
        for article in articles:
            title_element = article.find_next_sibling('a', class_='record_title')
            if title_element:
                title = title_element.text.strip()
                url = title_element['href']
                article_data[title] = url
        return article_data
