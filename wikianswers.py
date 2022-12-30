from helpers.wikiarticle import WikiArticle
from transformers import BartTokenizer, TFBartForConditionalGeneration, BertTokenizer, TFBertForQuestionAnswering
import tensorflow as tf
import argparse

import logging
logger = logging.getLogger(__name__)

class Summarizer:
    def __init__(self):
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        self.model = TFBartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

    def summarize(self, article: WikiArticle):
        inputs = self.tokenizer(article.content, return_tensors="tf", max_length=1024, truncation=True)
        summary_ids = self.model.generate(inputs["input_ids"], num_beams=4, max_length=100)
        return self.tokenizer.batch_decode(summary_ids, skip_special_tokens=True)

class QuestionAnswer:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("ydshieh/bert-base-cased-squad2")
        self.model = TFBertForQuestionAnswering.from_pretrained("ydshieh/bert-base-cased-squad2")

    def answer_question(self, article: WikiArticle, question):
        answer = "sorry, no answer found"
        context = None
        max_score = None
        for paragraph in article.get_article_by_paragraphs():
            max_output_length = 1000 if len(paragraph) > 1000 else len(paragraph)
            inputs = self.tokenizer(question, paragraph[0:max_output_length], return_tensors="tf")
            outputs = self.model(**inputs)
            answer_start_index = int(tf.math.argmax(outputs.start_logits, axis=-1)[0])
            answer_end_index = int(tf.math.argmax(outputs.end_logits, axis=-1)[0])
            predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
            decoded = self.tokenizer.decode(predict_answer_tokens)
            
            if decoded not in ('[CLS]', '[SEP]') and decoded != '':
                start_logit_max = tf.math.reduce_max(outputs.start_logits, axis=-1)
                end_logit_max = tf.math.reduce_max(outputs.end_logits, axis=-1)

                score = start_logit_max + end_logit_max

                logger.debug(f"computed score: {score}")
                logger.debug(f"value: {self.tokenizer.decode(predict_answer_tokens)}")

                if max_score == None or (start_logit_max + end_logit_max) > max_score:
                    max_score = start_logit_max + end_logit_max
                    logger.debug(f"potential answer: {self.tokenizer.decode(predict_answer_tokens)}")
                    answer = self.tokenizer.decode(predict_answer_tokens)
                    context = paragraph

        return (answer, context)

def main(verbose=False):
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    summarizer = Summarizer()
    question_answer = QuestionAnswer()

    while True:
        query = input("Enter a search query: ")
        if query == "--exit":
            break

        logger.info("searching...")
        
        article = WikiArticle(query)
        if article.content is None:
            logger.error("No article found")
            return

        logger.info(article.title)
        logger.info(article.url)


        logger.info("summarizing...")

        summary = summarizer.summarize(article)
        logger.info("=====================================")
        logger.info(summary)

        logger.info("=====================================")

        while True:
            question = input("Enter a question: ")
            if question == "--exit":
                break

            logger.info("answering...")
            response = question_answer.answer_question(article, question)
            logger.info("answer: " + response[0])
            if len(response) > 1:
                logger.info("context: " + response[1])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true", default=False)
    args = parser.parse_args()
    verbose = args.verbose

    main(args.verbose)