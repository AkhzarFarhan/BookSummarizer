# alternative_pdf_readers.py
"""
Alternative PDF reading implementations for different use cases
Choose the best one based on your specific needs
"""

import fitz  # PyMuPDF
import pdfplumber
from PyPDF2 import PdfReader
from typing import List, Tuple
import logging

class PyMuPDFReader:
    """
    Fastest and most reliable PDF reader
    Best for: Most use cases, especially complex PDFs
    """
    
    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> List[Tuple[int, str]]:
        """Extract text using PyMuPDF (fastest option)"""
        try:
            doc = fitz.open(pdf_path)
            pages_text = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                
                if text.strip():
                    pages_text.append((page_num + 1, text))
            
            doc.close()
            return pages_text
            
        except Exception as e:
            logging.error(f"PyMuPDF error: {e}")
            return []

class PdfPlumberReader:
    """
    Best for complex layouts and tables
    Best for: PDFs with tables, complex formatting
    """
    
    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> List[Tuple[int, str]]:
        """Extract text using pdfplumber (best for complex layouts)"""
        try:
            pages_text = []
            
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    
                    if text and text.strip():
                        pages_text.append((page_num, text))
            
            return pages_text
            
        except Exception as e:
            logging.error(f"pdfplumber error: {e}")
            return []

class PyPDF2Reader:
    """
    Lightweight and widely compatible
    Best for: Simple PDFs, when you want minimal dependencies
    """
    
    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> List[Tuple[int, str]]:
        """Extract text using PyPDF2 (lightweight option)"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                pages_text = []
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        text = page.extract_text()
                        if text.strip():
                            pages_text.append((page_num, text))
                    except Exception as e:
                        logging.warning(f"Could not extract page {page_num}: {e}")
                        continue
                
                return pages_text
                
        except Exception as e:
            logging.error(f"PyPDF2 error: {e}")
            return []

class MultiReaderFallback:
    """
    Tries multiple readers in order of preference
    Best for: Maximum reliability across different PDF types
    """
    
    def __init__(self):
        self.readers = [
            ("PyMuPDF", PyMuPDFReader.extract_text_from_pdf),
            ("pdfplumber", PdfPlumberReader.extract_text_from_pdf),
            ("PyPDF2", PyPDF2Reader.extract_text_from_pdf),
        ]
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Tuple[int, str]]:
        """Try multiple readers until one works"""
        for reader_name, reader_func in self.readers:
            try:
                print(f"Trying {reader_name}...")
                result = reader_func(pdf_path)
                
                if result:
                    print(f"✅ Successfully extracted text using {reader_name}")
                    return result
                else:
                    print(f"⚠️  {reader_name} returned no text")
                    
            except ImportError:
                print(f"❌ {reader_name} not installed")
                continue
            except Exception as e:
                print(f"❌ {reader_name} failed: {e}")
                continue
        
        print("❌ All PDF readers failed")
        return []

# Usage example for the main class
class EnhancedPDFChapterSummarizer:
    """Enhanced version with configurable PDF readers"""
    
    def __init__(self, gemini_api_key: str, reader_type: str = "auto"):
        """
        Initialize with configurable PDF reader
        
        Args:
            gemini_api_key: Your Gemini API key
            reader_type: "pymupdf", "pdfplumber", "pypdf2", or "auto"
        """
        # ... (same as before for Gemini setup)
        
        # Configure PDF reader
        if reader_type == "pymupdf":
            self.pdf_reader = PyMuPDFReader()
        elif reader_type == "pdfplumber":
            self.pdf_reader = PdfPlumberReader()
        elif reader_type == "pypdf2":
            self.pdf_reader = PyPDF2Reader()
        else:  # auto
            self.pdf_reader = MultiReaderFallback()
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Tuple[int, str]]:
        """Use the configured PDF reader"""
        return self.pdf_reader.extract_text_from_pdf(pdf_path)

# Installation commands for different readers:
"""
# For PyMuPDF (recommended - fastest and most reliable):
pip install PyMuPDF

# For pdfplumber (best for complex layouts):
pip install pdfplumber

# For PyPDF2 (lightweight):
pip install PyPDF2

# Install all for maximum compatibility:
pip install PyMuPDF pdfplumber PyPDF2
"""
