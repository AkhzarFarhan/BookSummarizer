# PDF Chapter Summarizer

A comprehensive Python tool for extracting and summarizing chapters from PDF books using AI APIs. This tool automatically detects chapter boundaries and generates detailed summaries for each chapter.

## 🌟 Features

- **Multiple PDF Readers**: Supports PyMuPDF, pdfplumber, and PyPDF2 with automatic fallback
- **Multiple AI APIs**: Works with Gemini, OpenAI, Claude, and Hugging Face models
- **Automatic Chapter Detection**: Intelligently identifies chapter boundaries
- **Cost Estimation**: Calculate processing costs before running
- **Multiple Output Formats**: JSON, plain text, and Markdown
- **Chunking Support**: Handles large chapters by splitting into manageable chunks
- **Command-Line Interface**: Easy-to-use CLI with comprehensive options

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the project files
git clone <your-repo-url>
cd pdf-chapter-summarizer

# Install dependencies
pip install -r requirements.txt

# For different PDF readers (choose based on your needs):
pip install PyMuPDF          # Recommended - fastest and most reliable
pip install pdfplumber       # Best for complex layouts and tables
pip install PyPDF2           # Lightweight option

# For different AI APIs:
pip install google-generativeai  # For Gemini (recommended - cheapest)
pip install openai               # For OpenAI GPT models
pip install anthropic            # For Claude
pip install transformers torch  # For Hugging Face (free, local processing)
```

### 2. Get API Keys

#### Gemini (Recommended - Most Cost-Effective)
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Cost: ~$0.00015 per 1K input tokens

#### OpenAI (High Quality)
1. Go to [OpenAI API](https://platform.openai.com/api-keys)
2. Create a new API key
3. Cost: GPT-3.5-turbo ~$0.001 per 1K tokens, GPT-4 ~$0.03 per 1K tokens

#### Claude (High Quality, Large Context)
1. Go to [Anthropic Console](https://console.anthropic.com/)
2. Create a new API key
3. Cost: Haiku ~$0.00025 per 1K input tokens

#### Hugging Face (Free)
- No API key required
- Runs locally on your machine
- Requires more computational resources

### 3. Basic Usage

```bash
# Using Gemini API (recommended)
python pdf_summarizer.py your_book.pdf --api gemini --key YOUR_GEMINI_API_KEY

# Using environment variable for API key
export GEMINI_API_KEY="your_api_key_here"
python pdf_summarizer.py your_book.pdf --api gemini

# Save results to file
python pdf_summarizer.py your_book.pdf --api gemini --key YOUR_KEY --output results.json

# Use different output format
python pdf_summarizer.py your_book.pdf --api gemini --key YOUR_KEY --output results.md --format markdown
```

## 📖 Detailed Usage

### Command-Line Options

```bash
python pdf_summarizer.py [PDF_FILE] [OPTIONS]

Required:
  PDF_FILE              Path to the PDF file to process

API Configuration:
  --api {gemini,openai,claude,huggingface}
                        AI API to use (default: gemini)
  --key API_KEY         API key (or use environment variable)
  --model MODEL_NAME    Model name (for OpenAI: gpt-3.5-turbo, gpt-4)

PDF Reader:
  --reader {auto,pymupdf,pdfplumber,pypdf2}
                        PDF reader to use (default: auto)

Output:
  --output, -o FILE     Output file path
  --format {json,txt,markdown}
                        Output format (default: json)
  --quiet, -q           Suppress progress output

Processing:
  --chunk-size SIZE     Max characters per chunk (default: 30000)
  --delay SECONDS       Delay between API calls (default: 2.0)
  --max-chapters N      Limit number of chapters to process

Utilities:
  --estimate-cost       Estimate cost without processing
  --list-chapters       List detected chapters without summarizing
```

### Examples

```bash
# Basic processing with Gemini
python pdf_summarizer.py book.pdf --api gemini --key YOUR_KEY

# High-quality processing with GPT-4
python pdf_summarizer.py book.pdf --api openai --key YOUR_KEY --model gpt-4

# Free processing with Hugging Face
python pdf_summarizer.py book.pdf --api huggingface

# Estimate cost before processing
python pdf_summarizer.py book.pdf --api gemini --key YOUR_KEY --estimate-cost

# List chapters without summarizing
python pdf_summarizer.py book.pdf --list-chapters

# Process only first 5 chapters
python pdf_summarizer.py book.pdf --api gemini --key YOUR_KEY --max-chapters 5

# Use specific PDF reader for complex layouts
python pdf_summarizer.py book.pdf --api gemini --key YOUR_KEY --reader pdfplumber

# Save as Markdown with custom settings
python pdf_summarizer.py book.pdf --api gemini --key YOUR_KEY \
  --output summary.md --format markdown --chunk-size 20000 --delay 1.5
```

## 💰 Cost Comparison

For processing a typical 100,000-token book:

| API | Cost | Quality | Speed | Notes |
|-----|------|---------|-------|--------|
| **Gemini Flash** | **$0.015** | High | Fast | **Recommended** - Best value |
| GPT-3.5-turbo | $0.10 | High | Fast | Good balance |
| GPT-4 | $3.00 | Very High | Medium | Premium quality |
| Claude Haiku | $0.025 | High | Fast | Good alternative |
| Claude Sonnet | $0.30 | Very High | Medium | High quality |
| Hugging Face | Free | Medium | Slow | Requires local compute |

## 🔧 Environment Variables

Create a `.env` file in the project directory:

```env
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_claude_api_key_here
DEFAULT_OUTPUT_DIR=./outputs/
DEFAULT_CHUNK_SIZE=30000
```

## 📁 Project Structure

```
pdf-chapter-summarizer/
├── pdf_summarizer.py          # Main CLI tool
├── pdf_chapter_summarizer.py  # Core summarizer class
├── api_alternatives.py        # Different AI API implementations
├── alternative_pdf_readers.py # Different PDF reader implementations
├── requirements.txt           # Python dependencies
├── .env.example              # Environment variable template
├── config.py                 # Configuration management
└── README.md                 # This file
```

## 🛠️ Troubleshooting

### Common Issues

1. **PDF text extraction fails**
   ```bash
   # Try different PDF readers
   python pdf_summarizer.py book.pdf --reader pymupdf
   python pdf_summarizer.py book.pdf --reader pdfplumber
   ```

2. **API rate limits**
   ```bash
   # Increase delay between calls
   python pdf_summarizer.py book.pdf --delay 5.0
   ```

3. **Memory issues with large PDFs**
   ```bash
   # Reduce chunk size
   python pdf_summarizer.py book.pdf --chunk-size 15000
   ```

4. **Chapter detection problems**
   - The tool uses pattern matching to detect chapters
   - For books with unusual chapter formatting, you may need to modify the patterns in the code

### PDF Reader Recommendations

- **PyMuPDF**: Best general-purpose option, handles most PDFs well
- **pdfplumber**: Best for PDFs with complex layouts, tables, or forms
- **PyPDF2**: Lightweight option, good for simple text-based PDFs

## 🎯 Advanced Usage

### Programmatic Usage

```python
from pdf_chapter_summarizer import PDFChapterSummarizer

# Initialize with your API key
summarizer = PDFChapterSummarizer("your_gemini_api_key")

# Process PDF
summaries = summarizer.process_pdf("book.pdf", "output.json")

# Print results
summarizer.print_results(summaries)
```

### Custom Chapter Detection

Modify the `chapter_patterns` in `pdf_chapter_summarizer.py`:

```python
self.chapter_patterns = [
    r'^\s*chapter\s+\d+',      # "Chapter 1"
    r'^\s*ch\s*\.\s*\d+',      # "Ch. 1"
    r'^\s*\d+\s*\.\s*[A-Z]',   # "1. Introduction"
    r'^\s*part\s+\d+',         # "Part 1"
    # Add your custom patterns here
]
```

## 📊 Output Formats

### JSON Output
```json
[
  {
    "chapter_number": 1,
    "chapter_title": "Introduction",
    "page_range": "Pages 1-15",
    "summary": "This chapter introduces...",
    "content_preview": "The first 200 characters..."
  }
]
```

### Markdown Output
```markdown
# Chapter Summaries

## Chapter 1: Introduction

**Pages:** 1-15

This chapter introduces the main concepts...

---
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

MIT License - feel free to use in your projects!

## 🆘 Support

- Create an issue on GitHub for bugs or feature requests
- Check the troubleshooting section for common problems
- Make sure your PDF contains extractable text (not just images)

## 🔮 Future Enhancements

- [ ] GUI interface
