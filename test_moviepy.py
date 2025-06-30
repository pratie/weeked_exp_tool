import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from wordcloud import WordCloud
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

# Load your data
df = pd.read_csv('your_file.csv')  # Replace with your actual file name
df.columns = df.columns.str.strip()

print("="*60)
print("COMPREHENSIVE FEEDBACK ANALYSIS FOR DECISION MAKING")
print("="*60)

# 1. FEEDBACK COVERAGE ANALYSIS
print("\n1. FEEDBACK COVERAGE ANALYSIS")
print("-" * 40)

total_records = len(df)
feedback_provided = df['Feedback'].notna().sum()
feedback_rate = feedback_provided / total_records

print(f"Total records: {total_records:,}")
print(f"Records with feedback: {feedback_provided:,}")
print(f"Feedback coverage rate: {feedback_rate:.2%}")

# Feedback rate by section/field
if 'GENESIS Section Name' in df.columns:
    section_feedback = df.groupby('GENESIS Section Name').agg({
        'Feedback': ['count', lambda x: x.notna().sum()]
    }).round(2)
    section_feedback.columns = ['total_records', 'feedback_count']
    section_feedback['feedback_rate'] = section_feedback['feedback_count'] / section_feedback['total_records']
    section_feedback = section_feedback.sort_values('feedback_rate', ascending=False)
    
    print("\nFeedback rate by section:")
    print(section_feedback)

# 2. SENTIMENT ANALYSIS
print("\n\n2. SENTIMENT ANALYSIS")
print("-" * 40)

def analyze_sentiment(text):
    """Analyze sentiment using TextBlob"""
    if pd.isna(text):
        return None, None
    try:
        blob = TextBlob(str(text))
        polarity = blob.sentiment.polarity  # -1 (negative) to 1 (positive)
        subjectivity = blob.sentiment.subjectivity  # 0 (objective) to 1 (subjective)
        
        if polarity > 0.1:
            sentiment = 'Positive'
        elif polarity < -0.1:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'
            
        return sentiment, polarity
    except:
        return None, None

# Apply sentiment analysis
feedback_data = df[df['Feedback'].notna()].copy()
if len(feedback_data) > 0:
    sentiments = feedback_data['Feedback'].apply(lambda x: analyze_sentiment(x))
    feedback_data['sentiment'] = [s[0] for s in sentiments]
    feedback_data['polarity_score'] = [s[1] for s in sentiments]
    
    # Sentiment distribution
    sentiment_counts = feedback_data['sentiment'].value_counts()
    print("Sentiment Distribution:")
    for sentiment, count in sentiment_counts.items():
        if sentiment:
            percentage = count / len(feedback_data) * 100
            print(f"  {sentiment}: {count} ({percentage:.1f}%)")
    
    # Average sentiment by section
    if 'GENESIS Section Name' in feedback_data.columns:
        section_sentiment = feedback_data.groupby('GENESIS Section Name').agg({
            'polarity_score': 'mean',
            'sentiment': lambda x: (x == 'Negative').sum()
        }).round(3)
        section_sentiment.columns = ['avg_sentiment_score', 'negative_feedback_count']
        section_sentiment = section_sentiment.sort_values('avg_sentiment_score')
        
        print("\nSentiment by section (most problematic first):")
        print(section_sentiment)

# 3. KEYWORD AND THEME ANALYSIS
print("\n\n3. KEYWORD AND THEME ANALYSIS")
print("-" * 40)

def extract_keywords(text_series, min_length=3, top_n=20):
    """Extract common keywords from feedback"""
    if text_series.empty:
        return []
    
    # Combine all text
    all_text = ' '.join(text_series.dropna().astype(str).str.lower())
    
    # Remove common words and extract meaningful terms
    common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'a', 'an', 'not', 'no'}
    
    # Extract words
    words = re.findall(r'\b[a-zA-Z]+\b', all_text)
    words = [word for word in words if len(word) >= min_length and word not in common_words]
    
    return Counter(words).most_common(top_n)

if len(feedback_data) > 0:
    keywords = extract_keywords(feedback_data['Feedback'])
    print("Most common keywords in feedback:")
    for i, (keyword, count) in enumerate(keywords[:15], 1):
        print(f"  {i:2d}. {keyword}: {count} mentions")

# 4. PROBLEM IDENTIFICATION
print("\n\n4. PROBLEM IDENTIFICATION")
print("-" * 40)

# Identify negative feedback patterns
if len(feedback_data) > 0:
    negative_feedback = feedback_data[feedback_data['sentiment'] == 'Negative']
    
    if len(negative_feedback) > 0:
        print(f"Found {len(negative_feedback)} negative feedback entries")
        
        # Most problematic fields
        if 'GENESIS Field Name' in negative_feedback.columns:
            problem_fields = negative_feedback['GENESIS Field Name'].value_counts().head(10)
            print("\nFields with most negative feedback:")
            for field, count in problem_fields.items():
                print(f"  {field}: {count} negative mentions")
        
        # Sample negative feedback
        print("\nSample negative feedback:")
        for i, feedback in enumerate(negative_feedback['Feedback'].head(5), 1):
            print(f"  {i}. \"{feedback[:100]}{'...' if len(str(feedback)) > 100 else ''}\"")

# 5. CORRELATION WITH PERFORMANCE METRICS
print("\n\n5. CORRELATION WITH PERFORMANCE METRICS")
print("-" * 40)

if len(feedback_data) > 0 and 'Confidence(%)' in feedback_data.columns:
    # Confidence vs sentiment
    conf_sentiment = feedback_data.groupby('sentiment')['Confidence(%)'].agg(['mean', 'count']).round(2)
    print("Confidence levels by sentiment:")
    print(conf_sentiment)
    
    # Accuracy vs sentiment
    if 'Accuracy(%)' in feedback_data.columns:
        acc_sentiment = feedback_data.groupby('sentiment')['Accuracy(%)'].agg(['mean', 'count']).round(2)
        print("\nAccuracy levels by sentiment:")
        print(acc_sentiment)

# 6. ACTIONABLE INSIGHTS AND RECOMMENDATIONS
print("\n\n6. ACTIONABLE INSIGHTS AND RECOMMENDATIONS")
print("-" * 40)

insights = []
recommendations = []

# Insight 1: Feedback coverage
if feedback_rate < 0.3:
    insights.append(f"LOW FEEDBACK COVERAGE: Only {feedback_rate:.1%} of users provided feedback")
    recommendations.append("Implement feedback prompts or incentives to increase user input")
elif feedback_rate > 0.7:
    insights.append(f"HIGH FEEDBACK ENGAGEMENT: {feedback_rate:.1%} of users provided feedback")
    recommendations.append("Leverage high engagement to gather more detailed user insights")

# Insight 2: Sentiment analysis
if len(feedback_data) > 0:
    negative_pct = sentiment_counts.get('Negative', 0) / len(feedback_data)
    if negative_pct > 0.3:
        insights.append(f"HIGH NEGATIVE SENTIMENT: {negative_pct:.1%} of feedback is negative")
        recommendations.append("Immediate attention needed to address user pain points")
    elif negative_pct < 0.1:
        insights.append(f"POSITIVE USER EXPERIENCE: Only {negative_pct:.1%} negative feedback")
        recommendations.append("Maintain current quality while scaling successful practices")

# Insight 3: Problem areas
if 'section_sentiment' in locals() and len(section_sentiment) > 0:
    worst_section = section_sentiment.index[0]
    worst_score = section_sentiment.iloc[0]['avg_sentiment_score']
    if worst_score < -0.2:
        insights.append(f"PROBLEMATIC SECTION: '{worst_section}' has sentiment score of {worst_score:.2f}")
        recommendations.append(f"Redesign or simplify the '{worst_section}' section urgently")

# Insight 4: Performance correlation
if len(feedback_data) > 0 and 'Confidence(%)' in feedback_data.columns:
    if 'sentiment' in feedback_data.columns:
        neg_conf = feedback_data[feedback_data['sentiment'] == 'Negative']['Confidence(%)'].mean()
        pos_conf = feedback_data[feedback_data['sentiment'] == 'Positive']['Confidence(%)'].mean()
        
        if neg_conf < pos_conf - 20:
            insights.append(f"CONFIDENCE GAP: Negative feedback users have {pos_conf - neg_conf:.1f}% lower confidence")
            recommendations.append("Focus on improving user guidance and interface clarity")

print("\nKEY INSIGHTS:")
for i, insight in enumerate(insights, 1):
    print(f"  {i}. {insight}")

print("\nRECOMMENDATIONS:")
for i, rec in enumerate(recommendations, 1):
    print(f"  {i}. {rec}")

# 7. VISUALIZATIONS
print("\n\n7. GENERATING VISUALIZATIONS...")
print("-" * 40)

if len(feedback_data) > 0:
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Sentiment distribution
    sentiment_counts.plot(kind='pie', ax=axes[0,0], autopct='%1.1f%%')
    axes[0,0].set_title('Feedback Sentiment Distribution')
    axes[0,0].set_ylabel('')
    
    # Feedback rate by section
    if 'section_feedback' in locals():
        section_feedback['feedback_rate'].plot(kind='bar', ax=axes[0,1])
        axes[0,1].set_title('Feedback Rate by Section')
        axes[0,1].set_xlabel('Section')
        axes[0,1].set_ylabel('Feedback Rate')
        axes[0,1].tick_params(axis='x', rotation=45)
    
    # Sentiment vs Confidence
    if 'sentiment' in feedback_data.columns and 'Confidence(%)' in feedback_data.columns:
        feedback_data.boxplot(column='Confidence(%)', by='sentiment', ax=axes[0,2])
        axes[0,2].set_title('Confidence by Sentiment')
        axes[0,2].set_xlabel('Sentiment')
        axes[0,2].set_ylabel('Confidence (%)')
    
    # Word cloud for all feedback
    try:
        all_feedback_text = ' '.join(feedback_data['Feedback'].dropna().astype(str))
        if len(all_feedback_text) > 0:
            wordcloud = WordCloud(width=400, height=300, background_color='white').generate(all_feedback_text)
            axes[1,0].imshow(wordcloud, interpolation='bilinear')
            axes[1,0].axis('off')
            axes[1,0].set_title('Feedback Word Cloud')
    except:
        axes[1,0].text(0.5, 0.5, 'Word Cloud\nUnavailable', ha='center', va='center')
        axes[1,0].set_title('Feedback Word Cloud')
    
    # Sentiment over time (if timestamp available)
    if 'Deleted Timestamp' in feedback_data.columns:
        try:
            feedback_data['timestamp'] = pd.to_datetime(feedback_data['Deleted Timestamp'])
            daily_sentiment = feedback_data.groupby([feedback_data['timestamp'].dt.date, 'sentiment']).size().unstack(fill_value=0)
            daily_sentiment.plot(ax=axes[1,1])
            axes[1,1].set_title('Sentiment Trends Over Time')
            axes[1,1].set_xlabel('Date')
            axes[1,1].set_ylabel('Count')
        except:
            axes[1,1].text(0.5, 0.5, 'Time Analysis\nUnavailable', ha='center', va='center')
    
    # Top keywords
    if keywords:
        keyword_df = pd.DataFrame(keywords[:10], columns=['keyword', 'count'])
        keyword_df.plot(x='keyword', y='count', kind='bar', ax=axes[1,2])
        axes[1,2].set_title('Top 10 Keywords in Feedback')
        axes[1,2].set_xlabel('Keywords')
        axes[1,2].set_ylabel('Frequency')
        axes[1,2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

# 8. EXPORT DETAILED ANALYSIS
print("\n8. EXPORTING DETAILED ANALYSIS...")
print("-" * 40)

# Create detailed feedback report
if len(feedback_data) > 0:
    feedback_report = feedback_data[['Case Number', 'GENESIS Section Name', 'GENESIS Field Name', 
                                   'Feedback', 'sentiment', 'polarity_score', 'Confidence(%)', 'Accuracy(%)']].copy()
    
    # Add priority flag
    feedback_report['priority'] = 'Low'
    feedback_report.loc[feedback_report['sentiment'] == 'Negative', 'priority'] = 'High'
    feedback_report.loc[(feedback_report['sentiment'] == 'Neutral') & 
                       (feedback_report['Confidence(%)'] < 50), 'priority'] = 'Medium'
    
    # Sort by priority and sentiment
    priority_order = {'High': 0, 'Medium': 1, 'Low': 2}
    feedback_report['priority_num'] = feedback_report['priority'].map(priority_order)
    feedback_report = feedback_report.sort_values(['priority_num', 'polarity_score'])
    feedback_report = feedback_report.drop('priority_num', axis=1)
    
    # Save to CSV
    feedback_report.to_csv('detailed_feedback_analysis.csv', index=False)
    print("✅ Detailed feedback analysis saved to 'detailed_feedback_analysis.csv'")
    
    # Summary statistics
    summary_stats = {
        'total_feedback': len(feedback_data),
        'negative_feedback': len(feedback_data[feedback_data['sentiment'] == 'Negative']),
        'positive_feedback': len(feedback_data[feedback_data['sentiment'] == 'Positive']),
        'avg_confidence': feedback_data['Confidence(%)'].mean(),
        'avg_sentiment_score': feedback_data['polarity_score'].mean(),
        'most_problematic_section': section_sentiment.index[0] if 'section_sentiment' in locals() and len(section_sentiment) > 0 else 'N/A'
    }
    
    print("\nSUMMARY STATISTICS:")
    for key, value in summary_stats.items():
        print(f"  {key}: {value}")

print("\n" + "="*60)
print("FEEDBACK ANALYSIS COMPLETE!")
print("="*60)
print("\nFiles generated:")
print("1. detailed_feedback_analysis.csv - Prioritized feedback with sentiment scores")
print("2. Multiple visualizations showing feedback patterns")
print("\nNext steps:")
print("1. Review high-priority feedback items first")
print("2. Address sections with lowest sentiment scores")
print("3. Implement changes based on specific user suggestions")
print("4. Set up regular feedback monitoring")

------------------------------------------------------------------------------------------------------------------



# 2. ACCURACY ANALYSIS BY SECTION
print("\n\n2. ACCURACY ANALYSIS BY SECTION")
print("-" * 40)

if 'Accuracy(%)' in df.columns and 'GENESIS Section Name' in df.columns:
    section_accuracy = df.groupby('GENESIS Section Name').agg({
        'Accuracy(%)': ['mean', 'std', 'count', 'min', 'max']
    }).round(2)
    
    # Flatten column names
    section_accuracy.columns = ['avg_accuracy', 'std_accuracy', 'record_count', 'min_accuracy', 'max_accuracy']
    section_accuracy = section_accuracy.sort_values('avg_accuracy', ascending=False)
    
    print("Accuracy statistics by section (best performing first):")
    print(section_accuracy)
    
    # Identify sections with low accuracy
    low_accuracy_threshold = 70  # You can adjust this threshold
    low_accuracy_sections = section_accuracy[section_accuracy['avg_accuracy'] < low_accuracy_threshold]
    
    if len(low_accuracy_sections) > 0:
        print(f"\n⚠️  SECTIONS WITH ACCURACY BELOW {low_accuracy_threshold}%:")
        for section, row in low_accuracy_sections.iterrows():
            print(f"  • {section}: {row['avg_accuracy']:.1f}% avg accuracy ({row['record_count']} records)")
    
    # Accuracy visualization
    plt.figure(figsize=(14, 8))
    
    plt.subplot(2, 2, 1)
    section_accuracy['avg_accuracy'].plot(kind='bar', color='skyblue')
    plt.title('Average Accuracy by Section')
    plt.xlabel('Section')
    plt.ylabel('Average Accuracy (%)')
    plt.xticks(rotation=45)
    plt.axhline(y=low_accuracy_threshold, color='red', linestyle='--', alpha=0.7, label=f'{low_accuracy_threshold}% threshold')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    section_accuracy['record_count'].plot(kind='bar', color='lightgreen')
    plt.title('Number of Records by Section')
    plt.xlabel('Section')
    plt.ylabel('Record Count')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 2, 3)
    df['Accuracy(%)'].hist(bins=20, alpha=0.7, color='orange')
    plt.title('Overall Accuracy Distribution')
    plt.xlabel('Accuracy (%)')
    plt.ylabel('Frequency')
    plt.axvline(x=df['Accuracy(%)'].mean(), color='red', linestyle='--', label=f'Mean: {df["Accuracy(%)"].mean():.1f}%')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    # Box plot of accuracy by section (top 8 sections by record count)
    top_sections = section_accuracy.nlargest(8, 'record_count').index
    section_data = df[df['GENESIS Section Name'].isin(top_sections)]
    section_data.boxplot(column='Accuracy(%)', by='GENESIS Section Name', ax=plt.gca())
    plt.title('Accuracy Distribution by Top Sections')
    plt.xlabel('Section')
    plt.ylabel('Accuracy (%)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

else:
    print("Accuracy or Section columns not found in the dataset")

# 3. CONFIDENCE vs ACCURACY ANALYSIS BY SECTION
print("\n\n3. CONFIDENCE vs ACCURACY ANALYSIS BY SECTION")
print("-" * 40)

if all(col in df.columns for col in ['Confidence(%)', 'Accuracy(%)', 'GENESIS Section Name']):
    conf_acc_analysis = df.groupby('GENESIS Section Name').agg({
        'Confidence(%)': ['mean', 'std'],
        'Accuracy(%)': ['mean', 'std'],
    }).round(2)
    
    # Flatten column names
    conf_acc_analysis.columns = ['avg_confidence', 'std_confidence', 'avg_accuracy', 'std_accuracy']
    
    # Calculate confidence-accuracy gap
    conf_acc_analysis['conf_acc_gap'] = conf_acc_analysis['avg_confidence'] - conf_acc_analysis['avg_accuracy']
    conf_acc_analysis = conf_acc_analysis.sort_values('conf_acc_gap', ascending=False)
    
    print("Confidence vs Accuracy by Section:")
    print("(Positive gap = overconfident, Negative gap = underconfident)")
    print(conf_acc_analysis)
    
    # Identify problematic patterns
    overconfident_sections = conf_acc_analysis[conf_acc_analysis['conf_acc_gap'] > 10]
    underconfident_sections = conf_acc_analysis[conf_acc_analysis['conf_acc_gap'] < -10]
    
    if len(overconfident_sections) > 0:
        print(f"\n🔴 OVERCONFIDENT SECTIONS (confidence > accuracy by >10%):")
        for section, row in overconfident_sections.iterrows():
            print(f"  • {section}: {row['conf_acc_gap']:.1f}% gap (Conf: {row['avg_confidence']:.1f}%, Acc: {row['avg_accuracy']:.1f}%)")
    
    if len(underconfident_sections) > 0:
        print(f"\n🟡 UNDERCONFIDENT SECTIONS (accuracy > confidence by >10%):")
        for section, row in underconfident_sections.iterrows():
            print(f"  • {section}: {row['conf_acc_gap']:.1f}% gap (Conf: {row['avg_confidence']:.1f}%, Acc: {row['avg_accuracy']:.1f}%)")
    
    # Visualization
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.scatter(conf_acc_analysis['avg_confidence'], conf_acc_analysis['avg_accuracy'], alpha=0.7)
    plt.plot([0, 100], [0, 100], 'r--', alpha=0.5, label='Perfect correlation')
    plt.xlabel('Average Confidence (%)')
    plt.ylabel('Average Accuracy (%)')
    plt.title('Confidence vs Accuracy by Section')
    plt.legend()
    
    # Add section labels to points
    for i, section in enumerate(conf_acc_analysis.index):
        plt.annotate(section[:15], (conf_acc_analysis.iloc[i]['avg_confidence'], 
                                   conf_acc_analysis.iloc[i]['avg_accuracy']), 
                    fontsize=8, alpha=0.7)
    
    plt.subplot(2, 2, 2)
    conf_acc_analysis['conf_acc_gap'].plot(kind='bar', color=['red' if x > 10 else 'yellow' if x < -10 else 'green' for x in conf_acc_analysis['conf_acc_gap']])
    plt.title('Confidence-Accuracy Gap by Section')
    plt.xlabel('Section')
    plt.ylabel('Gap (Confidence - Accuracy)')
    plt.xticks(rotation=45)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='Overconfident threshold')
    plt.axhline(y=-10, color='orange', linestyle='--', alpha=0.5, label='Underconfident threshold')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

print("\n" + "="*60)
print("ACCURACY ANALYSIS COMPLETE!")
print("="*60)
print("\nKey Findings:")
print("1. Section-wise accuracy performance")
print("2. Confidence vs accuracy alignment")
print("3. Sections requiring attention for accuracy improvement")
print("4. Over/under-confident sections identified")



##################################################################################################


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data
try:
    nltk.download('stopwords', quiet=True)
    print("✅ NLTK ready")
except:
    print("⚠️ Install NLTK: pip install nltk")

# Load data
df = pd.read_csv('your_file.csv')
df.columns = df.columns.str.strip()

print("="*80)
print("USER JUSTIFICATION TEXT - PURE INSIGHTS ANALYSIS")
print("="*80)

# Extract justification texts
justifications = df['User Justification Free Text'].dropna().astype(str).tolist()
print(f"📊 Analyzing {len(justifications)} user justifications")

if len(justifications) < 10:
    print("⚠️ Need at least 10 justifications for meaningful analysis")
    exit()

# Clean and prepare texts
def clean_text(text):
    """Basic text cleaning"""
    text = str(text).lower()
    text = ' '.join(text.split())  # Remove extra whitespace
    return text

cleaned_texts = [clean_text(text) for text in justifications if len(text.strip()) > 10]
print(f"📝 {len(cleaned_texts)} texts after cleaning")

# =============================================================================
# 1. TOPIC MODELING (LDA) - DISCOVER HIDDEN THEMES
# =============================================================================
print("\n\n🔍 1. TOPIC MODELING - DISCOVERING HIDDEN USER THEMES")
print("="*60)

def discover_topics(texts, n_topics=5):
    """Discover hidden topics using LDA"""
    
    # Get stopwords
    try:
        stop_words = list(stopwords.words('english'))
    except:
        stop_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
    
    # Add custom domain stopwords
    custom_stops = ['field', 'form', 'section', 'information', 'data', 'text', 'input', 'system']
    stop_words.extend(custom_stops)
    
    # Create document-term matrix
    vectorizer = CountVectorizer(
        max_features=100,
        stop_words=stop_words,
        ngram_range=(1, 2),  # Words and word-pairs
        min_df=2,  # Must appear in at least 2 texts
        max_df=0.8  # Remove words in >80% of texts
    )
    
    doc_term_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    # Perform LDA topic modeling
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=100
    )
    
    lda_result = lda.fit_transform(doc_term_matrix)
    
    return lda, lda_result, feature_names, vectorizer

# Determine optimal number of topics
n_topics = min(6, max(3, len(cleaned_texts) // 8))
print(f"🎯 Discovering {n_topics} hidden topics...")

lda_model, topic_distributions, feature_names, vectorizer = discover_topics(cleaned_texts, n_topics)

# Display discovered topics
print(f"\n🤖 DISCOVERED {n_topics} HIDDEN THEMES:")
print("-" * 50)

topic_themes = []
for topic_idx, topic in enumerate(lda_model.components_):
    # Get top words for this topic
    top_word_indices = topic.argsort()[-8:][::-1]
    top_words = [feature_names[i] for i in top_word_indices]
    
    topic_themes.append({
        'topic_num': topic_idx + 1,
        'keywords': top_words,
        'theme': ', '.join(top_words[:4])
    })
    
    print(f"📌 TOPIC {topic_idx + 1}: {', '.join(top_words[:4])}")
    print(f"   Keywords: {', '.join(top_words)}")

# Assign each justification to its dominant topic
dominant_topics = np.argmax(topic_distributions, axis=1)

# Show distribution of topics
topic_counts = pd.Series(dominant_topics).value_counts().sort_index()
print(f"\n📈 TOPIC DISTRIBUTION:")
for topic_idx, count in topic_counts.items():
    percentage = count / len(dominant_topics) * 100
    theme = topic_themes[topic_idx]['theme']
    print(f"   Topic {topic_idx + 1} ({theme}): {count} texts ({percentage:.1f}%)")

# Sample texts for each topic
print(f"\n📝 SAMPLE JUSTIFICATIONS BY TOPIC:")
for topic_idx in range(n_topics):
    topic_texts = [cleaned_texts[i] for i in range(len(cleaned_texts)) if dominant_topics[i] == topic_idx]
    if topic_texts:
        sample = topic_texts[0]
        theme = topic_themes[topic_idx]['theme']
        print(f"\n🔸 Topic {topic_idx + 1} ({theme}):")
        print(f"   \"{sample[:150]}{'...' if len(sample) > 150 else ''}\"")

# Visualize topics
plt.figure(figsize=(15, 10))

# Topic distribution
plt.subplot(2, 3, 1)
topic_counts.plot(kind='bar', color='skyblue', alpha=0.8)
plt.title('Distribution of User Topics', fontweight='bold')
plt.xlabel('Topic Number')
plt.ylabel('Number of Justifications')
plt.xticks(rotation=0)

# Topic word importance heatmap
plt.subplot(2, 3, 2)
topic_word_matrix = lda_model.components_[:, :20]  # Top 20 words
sns.heatmap(topic_word_matrix, cmap='Blues', cbar=True)
plt.title('Topic-Word Importance Matrix', fontweight='bold')
plt.xlabel('Top Words')
plt.ylabel('Topics')

plt.tight_layout()
plt.show()

# =============================================================================
# 2. TF-IDF ANALYSIS - DISTINCTIVE PHRASES
# =============================================================================
print("\n\n🔤 2. TF-IDF ANALYSIS - DISTINCTIVE PHRASES")
print("="*60)

def extract_distinctive_phrases(texts, max_features=30):
    """Extract most distinctive phrases using TF-IDF"""
    
    try:
        stop_words = list(stopwords.words('english'))
    except:
        stop_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
    
    custom_stops = ['field', 'form', 'section', 'information', 'data']
    stop_words.extend(custom_stops)
    
    # TF-IDF vectorizer
    tfidf = TfidfVectorizer(
        max_features=max_features,
        stop_words=stop_words,
        ngram_range=(1, 2),  # Single words and bigrams
        min_df=2,
        max_df=0.7
    )
    
    tfidf_matrix = tfidf.fit_transform(texts)
    feature_names = tfidf.get_feature_names_out()
    
    # Calculate average TF-IDF scores
    mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
    phrase_scores = list(zip(feature_names, mean_scores))
    phrase_scores.sort(key=lambda x: x[1], reverse=True)
    
    return phrase_scores, tfidf_matrix, tfidf

# Extract distinctive phrases
print("🔍 Extracting most distinctive phrases...")
distinctive_phrases, tfidf_matrix, tfidf_vectorizer = extract_distinctive_phrases(cleaned_texts)

print(f"\n💎 MOST DISTINCTIVE PHRASES:")
print("-" * 40)
for i, (phrase, score) in enumerate(distinctive_phrases[:15], 1):
    print(f"{i:2d}. '{phrase}' (score: {score:.3f})")

# Find phrases unique to each topic
print(f"\n🎯 DISTINCTIVE PHRASES BY TOPIC:")
print("-" * 40)

for topic_idx in range(n_topics):
    # Get texts for this topic
    topic_texts = [cleaned_texts[i] for i in range(len(cleaned_texts)) if dominant_topics[i] == topic_idx]
    
    if len(topic_texts) >= 3:  # Need minimum texts
        topic_phrases, _, _ = extract_distinctive_phrases(topic_texts, max_features=10)
        theme = topic_themes[topic_idx]['theme']
        
        print(f"\n📌 Topic {topic_idx + 1} ({theme}):")
        for phrase, score in topic_phrases[:5]:
            print(f"   • '{phrase}' ({score:.3f})")

# Visualize TF-IDF results
plt.figure(figsize=(14, 8))

# Overall distinctive phrases
plt.subplot(2, 2, 1)
top_phrases = distinctive_phrases[:12]
phrases, scores = zip(*top_phrases)
plt.barh(range(len(phrases)), scores, color='coral', alpha=0.8)
plt.yticks(range(len(phrases)), [p[:25] + '...' if len(p) > 25 else p for p in phrases])
plt.title('Most Distinctive Phrases', fontweight='bold')
plt.xlabel('TF-IDF Score')

# Phrase length distribution
plt.subplot(2, 2, 2)
phrase_lengths = [len(phrase.split()) for phrase, _ in distinctive_phrases[:20]]
plt.hist(phrase_lengths, bins=5, alpha=0.7, color='lightgreen', edgecolor='black')
plt.title('Phrase Length Distribution', fontweight='bold')
plt.xlabel('Number of Words in Phrase')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# =============================================================================
# 3. SEMANTIC SIMILARITY CLUSTERING
# =============================================================================
print("\n\n🧩 3. SEMANTIC SIMILARITY CLUSTERING")
print("="*60)

def semantic_clustering(texts, tfidf_matrix, n_clusters=5):
    """Group similar justifications using semantic clustering"""
    
    # Calculate cosine similarity matrix
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    # Perform K-means clustering on TF-IDF vectors
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(tfidf_matrix.toarray())
    
    return clusters, similarity_matrix, kmeans

# Determine number of clusters
n_clusters = min(5, max(3, len(cleaned_texts) // 10))
print(f"🎯 Grouping into {n_clusters} semantic clusters...")

clusters, similarity_matrix, kmeans_model = semantic_clustering(cleaned_texts, tfidf_matrix, n_clusters)

# Analyze clusters
print(f"\n🔗 SEMANTIC CLUSTERS (Similar Reasoning Patterns):")
print("-" * 55)

cluster_analysis = []
for cluster_id in range(n_clusters):
    # Get texts in this cluster
    cluster_texts = [cleaned_texts[i] for i in range(len(cleaned_texts)) if clusters[i] == cluster_id]
    cluster_size = len(cluster_texts)
    
    if cluster_size >= 2:
        # Extract representative phrases for this cluster
        cluster_tfidf = TfidfVectorizer(
            stop_words='english',
            max_features=8,
            ngram_range=(1, 2),
            min_df=1
        )
        
        try:
            cluster_matrix = cluster_tfidf.fit_transform(cluster_texts)
            cluster_features = cluster_tfidf.get_feature_names_out()
            cluster_scores = np.mean(cluster_matrix.toarray(), axis=0)
            top_cluster_words = [cluster_features[i] for i in cluster_scores.argsort()[-4:][::-1]]
        except:
            top_cluster_words = ['no distinctive words']
        
        cluster_analysis.append({
            'cluster_id': cluster_id + 1,
            'size': cluster_size,
            'keywords': top_cluster_words,
            'sample_text': cluster_texts[0] if cluster_texts else ""
        })
        
        print(f"\n🔸 CLUSTER {cluster_id + 1} ({cluster_size} similar justifications):")
        print(f"   Common themes: {', '.join(top_cluster_words)}")
        print(f"   Sample: \"{cluster_texts[0][:120]}{'...' if len(cluster_texts[0]) > 120 else ''}\"")

# Cluster distribution
cluster_counts = pd.Series(clusters).value_counts().sort_index()
print(f"\n📊 CLUSTER DISTRIBUTION:")
for cluster_id, count in cluster_counts.items():
    percentage = count / len(clusters) * 100
    analysis = next((c for c in cluster_analysis if c['cluster_id'] == cluster_id + 1), None)
    keywords = ', '.join(analysis['keywords'][:2]) if analysis else 'unknown'
    print(f"   Cluster {cluster_id + 1} ({keywords}): {count} texts ({percentage:.1f}%)")

# Visualize clustering results
plt.figure(figsize=(16, 10))

# Cluster size distribution
plt.subplot(2, 3, 1)
cluster_counts.plot(kind='bar', color='lightblue', alpha=0.8)
plt.title('Semantic Cluster Distribution', fontweight='bold')
plt.xlabel('Cluster Number')
plt.ylabel('Number of Justifications')
plt.xticks(rotation=0)

# Similarity heatmap (sample)
plt.subplot(2, 3, 2)
if len(similarity_matrix) <= 50:  # Only for small datasets
    sns.heatmap(similarity_matrix[:20, :20], cmap='Blues', cbar=True)
    plt.title('Text Similarity Matrix (Sample)', fontweight='bold')
else:
    plt.text(0.5, 0.5, 'Similarity Matrix\n(too large to display)', 
             ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('Text Similarity Matrix', fontweight='bold')

# Topic vs Cluster comparison
plt.subplot(2, 3, 3)
if len(dominant_topics) == len(clusters):
    topic_cluster_df = pd.DataFrame({
        'Topic': dominant_topics,
        'Cluster': clusters
    })
    crosstab = pd.crosstab(topic_cluster_df['Topic'], topic_cluster_df['Cluster'])
    sns.heatmap(crosstab, annot=True, fmt='d', cmap='Greens')
    plt.title('Topics vs Clusters Overlap', fontweight='bold')
    plt.xlabel('Semantic Cluster')
    plt.ylabel('LDA Topic')

plt.tight_layout()
plt.show()

# =============================================================================
# COMPREHENSIVE INSIGHTS SUMMARY
# =============================================================================
print("\n\n🎯 COMPREHENSIVE INSIGHTS SUMMARY")
print("="*60)

# Create results dataframe
results_df = pd.DataFrame({
    'justification_text': cleaned_texts,
    'dominant_topic': dominant_topics,
    'semantic_cluster': clusters
})

# Add topic themes
results_df['topic_theme'] = results_df['dominant_topic'].map(
    {i: theme['theme'] for i, theme in enumerate(topic_themes)}
)

# Add cluster info
results_df['cluster_keywords'] = results_df['semantic_cluster'].map(
    {(c['cluster_id']-1): ', '.join(c['keywords'][:2]) for c in cluster_analysis}
)

print(f"📋 ANALYSIS RESULTS:")
print(f"   • {len(cleaned_texts)} justifications analyzed")
print(f"   • {n_topics} hidden topics discovered")
print(f"   • {len(distinctive_phrases)} distinctive phrases identified")
print(f"   • {n_clusters} semantic clusters found")

print(f"\n🔍 TOP INSIGHTS:")
print(f"1. Most common topic: Topic {topic_counts.idxmax() + 1} ({topic_themes[topic_counts.idxmax()]['theme']})")
print(f"2. Largest cluster: Cluster {cluster_counts.idxmax() + 1} ({cluster_counts.max()} similar justifications)")
print(f"3. Most distinctive phrase: '{distinctive_phrases[0][0]}'")

# Export results
results_df.to_csv('user_justification_insights.csv', index=False)
print(f"\n💾 Results exported to 'user_justification_insights.csv'")

print("\n" + "="*80)
print("USER JUSTIFICATION INSIGHTS ANALYSIS COMPLETE!")
print("="*80)
print("\n🎯 WHAT YOU DISCOVERED:")
print("1. Hidden topics users actually discuss (not what you expected)")
print("2. Most distinctive phrases that characterize user thinking")
print("3. Groups of users with similar reasoning patterns")
print("\n📈 BUSINESS VALUE:")
print("• Focus development on most discussed topics")
print("• Use distinctive phrases to improve UI copy")
print("• Tailor help content to different user reasoning patterns")
