import PyPDF2
from .basetool import BaseTool
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer  # You can choose another summarizer if preferred

class ReadPDF(BaseTool):
    name = "read_pdf"
    description = "Read and summarize PDF files"

    def _run(self, file_path: str, sentences_count: int = 5):  # Added sentences_count for summary length
        try:
            # Read the PDF
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                text = ""
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text()

            # Summarize the extracted text
            summary = self.summarize_text(text, sentences_count)

            return summary
        except Exception as e:
            return {"error": str(e)}

    def summarize_text(self, text: str, sentences_count: int):
        # Summarize using sumy
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LsaSummarizer()
        summary_sentences = summarizer(parser.document, sentences_count)

        # Combine summary sentences into a single string
        summary = " ".join(str(sentence) for sentence in summary_sentences)
        return summary

    def _arun(self, file_path: str):
        raise NotImplementedError("Async version not implemented.")

