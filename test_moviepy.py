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
