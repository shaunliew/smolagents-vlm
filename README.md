# Singapore Price Comparison Tool 🛍️

A Streamlit-based web application that leverages [smolagents with Vision Language Model (VLM)](https://huggingface.co/blog/smolagents-can-see) capabilities to compare product prices between FairPrice and Lazada in Singapore. This tool showcases the power of visual AI-powered web browsing, enabling autonomous navigation and accurate product identification through visual content analysis.

![smolagents vlm demo](./smolagents-vlm-demo.gif)

[Youtube Link](https://youtu.be/Wx4mIwrFpxc)

## Key Visual AI Features

- **Vision Language Model Integration**: Utilizes [smolagents](https://github.com/huggingface/smolagents/tree/main)' VLM support for understanding webpage layouts, product images, and visual elements
- **Dynamic Screenshot Analysis**: Captures and processes real-time webpage screenshots for informed decision making
- **Visual Navigation**: Intelligently interacts with web elements based on visual context
- **Autonomous Product Identification**: Accurately identifies products through visual matching

## Features

- Real-time price comparison between FairPrice and Lazada
- Automated web navigation and product search
- Handles dynamic web elements and popups
- Extracts product details including:
  - Current price
  - Original price
  - Promotional discounts
- Clean and intuitive Streamlit interface
- Automatic reCAPTCHA handling
- Smart error handling and recovery

## Prerequisites

- Python 3.10
- Chrome browser installed
- ChromeDriver matching your Chrome version

## Installation

1. Clone the repository:
```bash
git clone https://github.com/shaunliew/smolagents-vlm.git
cd smolagents-vlm
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Important Note About Dependencies

This project uses the `smolagents` library, which is currently in active development and updates frequently. To ensure stability:

1. Use the specific versions listed in `requirements.txt`:
```txt
smolagents==1.6.0
streamlit==1.41.1
selenium==4.28.1
helium==5.1.0
python-dotenv==1.0.1
pillow==11.1.0
```

2. Or pin the exact version when installing:

```bash
pip install smolagents==1.6.0
```

If you encounter compatibility issues, please check the latest `smolagents` documentation and update the code accordingly.

## Environment Setup

1. Create a `.env` file in the project root:

```env
FIREWORKS_API_KEY=your_fireworks_api_key
```

1. (Optional) For alternative models, uncomment and configure in the code:

```python
# OpenAI
OPENAI_API_KEY=your_openai_api_key

# Local model
# No API key needed for local models
```

## Usage

1. Start the Streamlit application:

```bash
streamlit run app.py
```

1. Enter a product name in the search box
2. Click "Compare Prices"
3. View the comparison results

## Known Limitations

- Website changes may require code updates
- CAPTCHA handling might need manual intervention in some cases
- Some products might not be found on both platforms
- Price extraction might fail for some complex product pages
