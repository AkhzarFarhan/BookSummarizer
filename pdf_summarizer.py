#!/usr/bin/env python3
"""
PDF Chapter Summarizer - Complete CLI Tool
A comprehensive tool for extracting and summarizing chapters from PDF books
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our classes (assuming they're in the same directory)
try:
    from pdf_chapter_summarizer import PDFChapterSummarizer, ChapterSummary
    from api_alternatives import ConfigurableSummarizer
    from alternative_pdf_readers import MultiReaderFallback
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Make sure all required files are in the same directory")
    sys.exit(1)

class CLIPDFSummarizer:
    """Command-line interface for PDF Chapter Summarizer"""
    
    def __init__(self):
        self.supported_apis = ["gemini", "openai", "claude", "huggingface"]
        self.supported_readers = ["auto", "pymupdf", "pdfplumber", "pypdf2"]
    
    def create_parser(self) -> argparse.ArgumentParser:
        """Create command-line argument parser"""
        parser = argparse.ArgumentParser(
            description="Extract and summarize chapters from PDF books using AI",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  python pdf_summarizer.py book.pdf --api gemini --key YOUR_API_KEY
  python pdf_summarizer.py book.pdf --api openai --key YOUR_KEY --model gpt-4
  python pdf_summarizer.py book.pdf --api huggingface --no-key-required
  python pdf_summarizer.py book.pdf --reader pymupdf --output results.json
            """
        )
        
        # Required arguments
        parser.add_argument("pdf_path", help="Path to the PDF file to process")
        
        # API configuration
        parser.add_argument("--api", choices=self.supported_apis, default="gemini",
                          help="AI API to use for summarization (default: gemini)")
        parser.add_argument("--key", help="API key (can also use environment variable)")
        parser.add_argument("--model", help="Model name (for OpenAI: gpt-3.5-turbo, gpt-4)")
        
        # PDF reader configuration
        parser.add_argument("--reader", choices=self.supported_readers, default="auto",
                          help="PDF reader to use (default: auto)")
        
        # Output configuration
        parser.add_argument("--output", "-o", help="Output file path (JSON format)")
        parser.add_argument("--format", choices=["json", "txt", "markdown"], default="json",
                          help="Output format (default: json)")
        parser.add_argument("--quiet", "-q", action="store_true",
                          help="Suppress progress output")
        
        # Processing options
        parser.add_argument("--chunk-size", type=int, default=30000,
                          help="Maximum characters per chunk (default: 30000)")
        parser.add_argument("--delay", type=float, default=2.0,
                          help="Delay between API calls in seconds (default: 2.0)")
        parser.add_argument("--max-chapters", type=int,
                          help="Maximum number of chapters to process")
        
        # Utility options
        parser.add_argument("--estimate-cost", action="store_true",
                          help="Estimate processing cost without actually processing")
        parser.add_argument("--list-chapters", action="store_true",
                          help="List detected chapters without summarizing")
        parser.add_argument("--version", action="version", version="PDF Chapter Summarizer 1.0.0")
        
        return parser
    
    def validate_args(self, args) -> bool:
        """Validate command-line arguments"""
        # Check if PDF file exists
        if not os.path.exists(args.pdf_path):
            logger.error(f"PDF file not found: {args.pdf_path}")
            return False
        
        # Check API key requirements
        if args.api != "huggingface" and not args.key:
            # Try to get from environment
            env_vars = {
                "gemini": "GEMINI_API_KEY",
                "openai": "OPENAI_API_KEY",
                "claude": "ANTHROPIC_API_KEY"
            }
            
            env_var = env_vars.get(args.api)
            if env_var:
                args.key = os.getenv(env_var)
            
            if not args.key:
                logger.error(f"API key required for {args.api}. Use --key or set {env_var} environment variable")
                return False
        
        return True
    
    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (1 token ≈ 4 characters)"""
        return len(text) // 4
    
    def process_pdf(self, args) -> List[ChapterSummary]:
        """Process PDF and return chapter summaries"""
        if not args.quiet:
            print(f"🔍 Processing PDF: {args.pdf_path}")
            print(f"📚 Using API: {args.api}")
            print(f"📖 Using PDF reader: {args.reader}")
        
        # Initialize PDF reader
        if args.reader == "auto":
            pdf_reader = MultiReaderFallback()
        else:
            # Use specific reader (implementation would need to be added)
            pdf_reader = MultiReaderFallback()  # Fallback for now
        
        # Extract text
        if not args.quiet:
            print("📄 Extracting text from PDF...")
        
        pages_text = pdf_reader.extract_text_from_pdf(args.pdf_path)
        if not pages_text:
            logger.error("Could not extract any text from PDF")
            return []
        
        if not args.quiet:
            print(f"✅ Extracted text from {len(pages_text)} pages")
        
        # Initialize summarizer (simplified version for CLI)
        if args.api == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=args.key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            def summarize_func(text, title=""):
                prompt = f"""
                Summarize this chapter comprehensively:
                Chapter: {title}
                Content: {text}
                
                Provide a detailed 200-400 word summary covering main ideas and key points.
                """
                try:
                    response = model.generate_content(prompt)
                    return response.text.strip()
                except Exception as e:
                    return f"Error: {str(e)}"
        
        elif args.api == "huggingface":
            try:
                from transformers import pipeline
                summarizer_model = pipeline("summarization", model="facebook/bart-large-cnn")
                
                def summarize_func(text, title=""):
                    max_len = min(len(text) // 4, 200)  # Rough token estimate
                    result = summarizer_model(text[:1024], max_length=max_len, min_length=50)
                    return result[0]['summary_text']
            except ImportError:
                logger.error("Transformers not installed. Run: pip install transformers torch")
                return []
        
        else:
            logger.error(f"API {args.api} not fully implemented in CLI version")
            return []
        
        # Detect chapters (simplified)
        if not args.quiet:
            print("🔍 Detecting chapters...")
        
        chapters = self.detect_chapters_simple(pages_text)
        
        if args.list_chapters:
            print("\n📋 Detected Chapters:")
            for i, chapter in enumerate(chapters, 1):
                print(f"{i:2}. {chapter['title'][:60]}... (Pages {chapter['start_page']}-{chapter['end_page']})")
            return []
        
        if not args.quiet:
            print(f"📚 Found {len(chapters)} chapters")
        
        # Limit chapters if specified
        if args.max_chapters:
            chapters = chapters[:args.max_chapters]
            if not args.quiet:
                print(f"🔢 Processing first {len(chapters)} chapters")
        
        # Estimate cost if requested
        if args.estimate_cost:
            total_chars = sum(len(''.join(ch['content'])) for ch in chapters)
            tokens = self.estimate_tokens(str(total_chars))
            
            cost_estimates = {
                "gemini": tokens * 0.00015 / 1000,
                "openai": tokens * 0.001 / 1000,
                "claude": tokens * 0.00025 / 1000,
                "huggingface": 0.0
            }
            
            print(f"\n💰 Cost Estimate:")
            print(f"Total characters: {total_chars:,}")
            print(f"Estimated tokens: {tokens:,}")
            print(f"Estimated cost for {args.api}: ${cost_estimates[args.api]:.4f}")
            return []
        
        # Process chapters
        summaries = []
        for i, chapter in enumerate(chapters, 1):
            if not args.quiet:
                print(f"⚙️  Processing Chapter {i}/{len(chapters)}: {chapter['title'][:40]}...")
            
            try:
                content = '\n'.join(chapter['content'])
                
                # Chunk if necessary
                if len(content) > args.chunk_size:
                    chunks = [content[i:i+args.chunk_size] for i in range(0, len(content), args.chunk_size)]
                    chunk_summaries = []
                    
                    for j, chunk in enumerate(chunks):
                        if not args.quiet:
                            print(f"  📝 Summarizing chunk {j+1}/{len(chunks)}...")
                        summary = summarize_func(chunk, f"{chapter['title']} (Part {j+1})")
                        chunk_summaries.append(summary)
                        time.sleep(args.delay)
                    
                    final_summary = summarize_func('\n'.join(chunk_summaries), chapter['title'])
                else:
                    final_summary = summarize_func(content, chapter['title'])
                
                summary_obj = ChapterSummary(
                    chapter_number=i,
                    chapter_title=chapter['title'],
                    content_preview=content[:200] + "..." if len(content) > 200 else content,
                    summary=final_summary,
                    page_range=(chapter['start_page'], chapter['end_page'])
                )
                
                summaries.append(summary_obj)
                
                if not args.quiet:
                    print(f"  ✅ Chapter {i} completed")
                
                time.sleep(args.delay)
                
            except Exception as e:
                logger.error(f"Error processing chapter {i}: {e}")
                continue
        
        return summaries
    
    def detect_chapters_simple(self, pages_text: List[Tuple[int, str]]) -> List[Dict]:
        """Simplified chapter detection"""
        import re
        
        chapters = []
        current_chapter = None
        
        chapter_patterns = [
            r'^\s*chapter\s+\d+',
            r'^\s*ch\s*\.\s*\d+',
            r'^\s*\d+\s*\.\s*[A-Z]',
            r'^\s*part\s+\d+',
        ]
        
        for page_num, text in pages_text:
            lines = text.split('\n')
            
            for line in lines:
                line_clean = line.strip().lower()
                
                for pattern in chapter_patterns:
                    if re.match(pattern, line_clean, re.IGNORECASE):
                        if current_chapter:
                            current_chapter['end_page'] = page_num - 1
                            chapters.append(current_chapter)
                        
                        current_chapter = {
                            'title': line.strip(),
                            'start_page': page_num,
                            'content': []
                        }
                        break
            
            if current_chapter:
                current_chapter['content'].append(text)
        
        if current_chapter:
            current_chapter['end_page'] = pages_text[-1][0]
            chapters.append(current_chapter)
        
        # If no chapters found, treat as single document
        if not chapters:
            all_content = [text for _, text in pages_text]
            chapters = [{
                'title': 'Complete Document',
                'start_page': pages_text[0][0] if pages_text else 1,
                'end_page': pages_text[-1][0] if pages_text else 1,
                'content': all_content
            }]
        
        return chapters
    
    def save_results(self, summaries: List[ChapterSummary], output_path: str, format_type: str):
        """Save results in specified format"""
        if format_type == "json":
            self.save_json(summaries, output_path)
        elif format_type == "txt":
            self.save_txt(summaries, output_path)
        elif format_type == "markdown":
            self.save_markdown(summaries, output_path)
    
    def save_json(self, summaries: List[ChapterSummary], output_path: str):
        """Save as JSON"""
        data = []
        for summary in summaries:
            data.append({
                'chapter_number': summary.chapter_number,
                'chapter_title': summary.chapter_title,
                'page_range': f"Pages {summary.page_range[0]}-{summary.page_range[1]}",
                'summary': summary.summary,
                'content_preview': summary.content_preview
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def save_txt(self, summaries: List[ChapterSummary], output_path: str):
        """Save as plain text"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("CHAPTER SUMMARIES\n")
            f.write("=" * 80 + "\n\n")
            
            for summary in summaries:
                f.write(f"Chapter {summary.chapter_number}: {summary.chapter_title}\n")
                f.write(f"Pages: {summary.page_range[0]}-{summary.page_range[1]}\n")
                f.write("-" * 60 + "\n")
                f.write(summary.summary + "\n")
                f.write("-" * 60 + "\n\n")
    
    def save_markdown(self, summaries: List[ChapterSummary], output_path: str):
        """Save as Markdown"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Chapter Summaries\n\n")
            
            for summary in summaries:
                f.write(f"## Chapter {summary.chapter_number}: {summary.chapter_title}\n\n")
                f.write(f"**Pages:** {summary.page_range[0]}-{summary.page_range[1]}\n\n")
                f.write(summary.summary + "\n\n")
                f.write("---\n\n")
    
    def print_results(self, summaries: List[ChapterSummary]):
        """Print results to console"""
        print("\n" + "="*80)
        print("CHAPTER SUMMARIES")
        print("="*80)
        
        for summary in summaries:
            print(f"\n📖 Chapter {summary.chapter_number}: {summary.chapter_title}")
            print(f"📄 Pages: {summary.page_range[0]}-{summary.page_range[1]}")
            print("-" * 60)
            print(summary.summary)
            print("-" * 60)
    
    def run(self):
        """Main entry point"""
        parser = self.create_parser()
        args = parser.parse_args()
        
        if not self.validate_args(args):
            sys.exit(1)
        
        try:
            summaries = self.process_pdf(args)
            
            if not summaries:
                if not (args.list_chapters or args.estimate_cost):
                    logger.error("No summaries generated")
                sys.exit(0)
            
            # Save results
            if args.output:
                self.save_results(summaries, args.output, args.format)
                if not args.quiet:
                    print(f"\n💾 Results saved to: {args.output}")
            
            # Print to console if not quiet and no output file
            if not args.quiet and not args.output:
                self.print_results(summaries)
            
            if not args.quiet:
                print(f"\n✅ Successfully processed {len(summaries)} chapters!")
            
        except KeyboardInterrupt:
            print("\n⚠️  Processing interrupted by user")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            sys.exit(1)

if __name__ == "__main__":
    cli = CLIPDFSummarizer()
    cli.run()
