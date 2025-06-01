import os
import re
import google.generativeai as genai
from pypdf2 import PdfReader
import PyPDF2
from typing import List, Dict, Tuple
import time
import json
from dataclasses import dataclass
from pathlib import Path

@dataclass
class ChapterSummary:
    chapter_number: int
    chapter_title: str
    content_preview: str
    summary: str
    page_range: Tuple[int, int]

class PDFChapterSummarizer:
    def __init__(self, gemini_api_key: str):
        """
        Initialize the PDF Chapter Summarizer with Gemini API
        
        Args:
            gemini_api_key (str): Your Google Gemini API key
        """
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')  # Using flash for cost efficiency
        
        # Common chapter patterns to detect chapter boundaries
        self.chapter_patterns = [
            r'^\s*chapter\s+\d+',
            r'^\s*ch\s*\.\s*\d+',
            r'^\s*\d+\s*\.\s*[A-Z]',
            r'^\s*part\s+\d+',
            r'^\s*section\s+\d+',
            r'^\s*[IVX]+\s*\.\s*[A-Z]',  # Roman numerals
        ]
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Tuple[int, str]]:
        """
        Extract text from PDF with page numbers
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            List[Tuple[int, str]]: List of (page_number, text) tuples
        """
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                pages_text = []
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        text = page.extract_text()
                        if text.strip():  # Only add non-empty pages
                            pages_text.append((page_num, text))
                    except Exception as e:
                        print(f"Warning: Could not extract text from page {page_num}: {e}")
                        continue
                
                return pages_text
                
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return []
    
    def detect_chapters(self, pages_text: List[Tuple[int, str]]) -> List[Dict]:
        """
        Detect chapter boundaries in the extracted text
        
        Args:
            pages_text: List of (page_number, text) tuples
            
        Returns:
            List[Dict]: List of chapter information dictionaries
        """
        chapters = []
        current_chapter = None
        
        for page_num, text in pages_text:
            lines = text.split('\n')
            
            for line_idx, line in enumerate(lines):
                line_clean = line.strip().lower()
                
                # Check if line matches any chapter pattern
                for pattern in self.chapter_patterns:
                    if re.match(pattern, line_clean, re.IGNORECASE):
                        # Save previous chapter if exists
                        if current_chapter:
                            current_chapter['end_page'] = page_num - 1
                            chapters.append(current_chapter)
                        
                        # Extract chapter title (try to get the full title)
                        title_lines = [line.strip()]
                        # Look for continuation on next few lines
                        for next_line_idx in range(line_idx + 1, min(line_idx + 3, len(lines))):
                            next_line = lines[next_line_idx].strip()
                            if next_line and not re.match(r'^\d+$', next_line):
                                title_lines.append(next_line)
                            else:
                                break
                        
                        chapter_title = ' '.join(title_lines)
                        
                        current_chapter = {
                            'chapter_number': len(chapters) + 1,
                            'title': chapter_title,
                            'start_page': page_num,
                            'content': []
                        }
                        break
            
            # Add page content to current chapter
            if current_chapter:
                current_chapter['content'].append(text)
        
        # Don't forget the last chapter
        if current_chapter:
            current_chapter['end_page'] = pages_text[-1][0]
            chapters.append(current_chapter)
        
        # If no chapters detected, treat entire document as one chapter
        if not chapters:
            all_content = [text for _, text in pages_text]
            chapters = [{
                'chapter_number': 1,
                'title': 'Complete Document',
                'start_page': pages_text[0][0] if pages_text else 1,
                'end_page': pages_text[-1][0] if pages_text else 1,
                'content': all_content
            }]
        
        return chapters
    
    def chunk_text(self, text: str, max_chars: int = 30000) -> List[str]:
        """
        Split text into chunks that fit within API limits
        
        Args:
            text (str): Text to chunk
            max_chars (int): Maximum characters per chunk
            
        Returns:
            List[str]: List of text chunks
        """
        if len(text) <= max_chars:
            return [text]
        
        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) <= max_chars:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def summarize_text(self, text: str, chapter_title: str = "") -> str:
        """
        Summarize text using Gemini API
        
        Args:
            text (str): Text to summarize
            chapter_title (str): Chapter title for context
            
        Returns:
            str: Summary of the text
        """
        try:
            prompt = f"""
            Please provide a comprehensive summary of the following chapter content.
            Chapter Title: {chapter_title}
            
            Requirements for the summary:
            1. Capture the main ideas and key points
            2. Maintain logical flow and structure
            3. Include important details, examples, or concepts
            4. Keep it concise but informative (200-400 words)
            5. Use clear, readable language
            
            Content to summarize:
            {text}
            
            Summary:
            """
            
            response = self.model.generate_content(prompt)
            return response.text.strip()
            
        except Exception as e:
            print(f"Error generating summary: {e}")
            return f"Error: Could not generate summary for this chapter. {str(e)}"
    
    def process_chapter(self, chapter: Dict) -> ChapterSummary:
        """
        Process a single chapter to generate summary
        
        Args:
            chapter: Chapter dictionary from detect_chapters
            
        Returns:
            ChapterSummary: Processed chapter with summary
        """
        # Combine all content for this chapter
        full_content = '\n'.join(chapter['content'])
        
        # Create content preview (first 200 characters)
        content_preview = full_content[:200] + "..." if len(full_content) > 200 else full_content
        
        # Split into chunks if content is too long
        chunks = self.chunk_text(full_content)
        
        if len(chunks) == 1:
            # Single chunk - direct summarization
            summary = self.summarize_text(chunks[0], chapter['title'])
        else:
            # Multiple chunks - summarize each then combine
            chunk_summaries = []
            for i, chunk in enumerate(chunks):
                print(f"  Summarizing chunk {i+1}/{len(chunks)}...")
                chunk_summary = self.summarize_text(chunk, f"{chapter['title']} (Part {i+1})")
                chunk_summaries.append(chunk_summary)
                time.sleep(1)  # Rate limiting
            
            # Combine chunk summaries into final summary
            combined_summary = '\n\n'.join(chunk_summaries)
            final_prompt = f"""
            Please create a unified summary from these partial summaries of the chapter "{chapter['title']}":
            
            {combined_summary}
            
            Create a coherent, comprehensive summary that combines all the key points:
            """
            
            summary = self.summarize_text(final_prompt, chapter['title'])
        
        return ChapterSummary(
            chapter_number=chapter['chapter_number'],
            chapter_title=chapter['title'],
            content_preview=content_preview,
            summary=summary,
            page_range=(chapter['start_page'], chapter['end_page'])
        )
    
    def process_pdf(self, pdf_path: str, output_file: str = None) -> List[ChapterSummary]:
        """
        Main method to process PDF and generate chapter summaries
        
        Args:
            pdf_path (str): Path to the PDF file
            output_file (str): Optional path to save results as JSON
            
        Returns:
            List[ChapterSummary]: List of chapter summaries
        """
        print(f"Processing PDF: {pdf_path}")
        
        # Extract text from PDF
        print("Extracting text from PDF...")
        pages_text = self.extract_text_from_pdf(pdf_path)
        
        if not pages_text:
            print("Error: Could not extract any text from PDF")
            return []
        
        print(f"Extracted text from {len(pages_text)} pages")
        
        # Detect chapters
        print("Detecting chapters...")
        chapters = self.detect_chapters(pages_text)
        print(f"Found {len(chapters)} chapters")
        
        # Process each chapter
        chapter_summaries = []
        for i, chapter in enumerate(chapters, 1):
            print(f"Processing Chapter {i}: {chapter['title'][:50]}...")
            try:
                summary = self.process_chapter(chapter)
                chapter_summaries.append(summary)
                print(f"  ✓ Chapter {i} completed")
                time.sleep(2)  # Rate limiting for API calls
            except Exception as e:
                print(f"  ✗ Error processing Chapter {i}: {e}")
                continue
        
        # Save results if output file specified
        if output_file:
            self.save_results(chapter_summaries, output_file)
        
        return chapter_summaries
    
    def save_results(self, summaries: List[ChapterSummary], output_file: str):
        """
        Save results to JSON file
        
        Args:
            summaries: List of ChapterSummary objects
            output_file: Path to output file
        """
        results = []
        for summary in summaries:
            results.append({
                'chapter_number': summary.chapter_number,
                'chapter_title': summary.chapter_title,
                'page_range': f"Pages {summary.page_range[0]}-{summary.page_range[1]}",
                'content_preview': summary.content_preview,
                'summary': summary.summary
            })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to {output_file}")
    
    def print_results(self, summaries: List[ChapterSummary]):
        """
        Print results to console in a formatted way
        
        Args:
            summaries: List of ChapterSummary objects
        """
        print("\n" + "="*80)
        print("CHAPTER SUMMARIES")
        print("="*80)
        
        for summary in summaries:
            print(f"\n📖 Chapter {summary.chapter_number}: {summary.chapter_title}")
            print(f"📄 Pages: {summary.page_range[0]}-{summary.page_range[1]}")
            print("-" * 60)
            print(summary.summary)
            print("-" * 60)

# Example usage
def main():
    # Initialize with your Gemini API key
    API_KEY = "your-gemini-api-key-here"  # Replace with your actual API key
    
    summarizer = PDFChapterSummarizer(API_KEY)
    
    # Process PDF
    pdf_path = "your-book.pdf"  # Replace with your PDF path
    output_file = "chapter_summaries.json"
    
    try:
        summaries = summarizer.process_pdf(pdf_path, output_file)
        
        # Print results
        summarizer.print_results(summaries)
        
        print(f"\n✅ Successfully processed {len(summaries)} chapters!")
        print(f"📁 Results saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
