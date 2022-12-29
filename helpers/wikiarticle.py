import wikipedia
import logging

logger = logging.getLogger(__name__)

class WikiArticle:
    def __init__(self, query):
        self.query = query
        try:
            self.set_up(query)
        except wikipedia.exceptions.DisambiguationError as e:
            logger.error('Multiple possible articles found. Please select one of the following:')
            for (idx, val) in enumerate(e.options):
                logger.error(f"{idx}: \t{val}")
            selection = int(input(f"Enter the index of the article you want to select (0-{len(e.options)}):"))
            self.set_up(e.options[selection])
        except wikipedia.exceptions.PageError as e:
            logger.error(f"Could not find a page for {query}. Using auto-suggest instead.")
            self.set_up(query, auto_suggest=True)
        except Exception as e:
            logger.error(f"An error occurred: {e}")
    
    def set_up(self, query, auto_suggest=False):
        self.article = wikipedia.page(query, auto_suggest=auto_suggest)
        self.title = self.article.title
        self.content = self.article.content
        self.url = self.article.url

    def get_article_by_paragraphs(self):
        return self.content.split('\n\n')

    def __str__(self):
        return self.title

if __name__ == "__main__":
    query = input("Enter a search query: ")
    article = WikiArticle(query)
    logger.info(article.title)
