# Green Finance Optimization Platform

An AI-powered platform for evaluating, prioritizing, and optimizing green finance investments using BERT, T5, and UMAP models. This platform helps financial institutions allocate capital to the most impactful and sustainable projects through advanced ESG analysis.

## 🌟 Features

- **Advanced ESG Analysis**
  - BERT-based document understanding
  - T5-powered ESG scoring
  - UMAP dimension reduction for visualization

- **Data Processing**
  - Automated ESG data aggregation
  - Climate data integration
  - Financial metrics processing

- **Portfolio Optimization**
  - Risk assessment
  - Budget constraint handling
  - ESG impact maximization

## 📋 Prerequisites

```bash
Python 3.8+
PyTorch 1.9+
Transformers 4.0+
UMAP-learn
scikit-learn
numpy
pandas
```

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/green-finance-platform.git
cd green-finance-platform
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Usage

### BERT Analysis
```python
from models.bert_analyzer import BERTAnalyzer

analyzer = BERTAnalyzer()
features = analyzer.extract_features(texts)
```

### T5 Scoring
```python
from models.t5_analyzer import T5Analyzer

scorer = T5Analyzer()
esg_scores = scorer.get_esg_scores(texts)
```

### UMAP Dimension Reduction
```python
from models.umap_reducer import ESGDimensionReducer

reducer = ESGDimensionReducer()
embeddings = reducer.reduce_dimensions(bert_features, esg_scores)
```

## 📁 Project Structure

```
green-finance-platform/
├── models/
│   ├── bert_analyzer.py
│   ├── t5_analyzer.py
│   └── umap_reducer.py
├── requirements.txt
└── README.md
```

## 📊 Example Output

The platform provides:
- ESG scores (0-100)
- Risk assessments
- Portfolio recommendations
- Dimension-reduced visualizations

## 🙏 Acknowledgments

- Hugging Face for BERT and T5 models
- UMAP-learn developers
- The open-source community
