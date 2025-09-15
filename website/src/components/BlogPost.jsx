import React from 'react';
import { useParams, Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { ArrowLeft, Calendar, Clock, User, Tag, Share2 } from 'lucide-react';
import Card from './ui/Card';
import Badge from './ui/Badge';
import Button from './ui/Button';
import SEO from './SEO';
import PageTransition from './ui/PageTransition';
import AnimatedSection from './ui/AnimatedSection';

const BlogPost = () => {
  const { slug } = useParams();

  // Sample blog posts data - in a real app, this would come from a CMS or MDX files
  const blogPosts = {
    'building-modern-portfolio-react-tailwind': {
      id: 1,
      title: "Building a Modern Portfolio with React and Tailwind CSS",
      excerpt: "Learn how I built this portfolio using React, Tailwind CSS, and Framer Motion with a focus on performance and accessibility.",
      content: `
# Building a Modern Portfolio with React and Tailwind CSS

## Introduction

Building a personal portfolio is an essential step for any developer looking to showcase their skills and projects. In this post, I'll walk you through how I built this very portfolio using modern web technologies.

## Technology Stack

- **React 18**: For building the user interface with modern hooks
- **Tailwind CSS**: For utility-first styling and responsive design
- **Framer Motion**: For smooth animations and microinteractions
- **Vite**: For fast development and optimized builds
- **React Router**: For client-side routing

## Key Features

### 1. Performance Optimization
- Code splitting with lazy loading
- Optimized images and assets
- Minimal bundle size

### 2. Accessibility
- Proper semantic HTML
- ARIA labels and screen reader support
- Keyboard navigation
- Skip links for better UX

### 3. Responsive Design
- Mobile-first approach
- Fluid typography and spacing
- Touch-friendly interactions

### 4. SEO Optimization
- Meta tags and Open Graph data
- Structured data markup
- Proper heading hierarchy

## Implementation Details

Coming soon... This blog functionality is currently under development. Stay tuned for detailed implementation guides!

## Conclusion

Building a modern portfolio requires careful consideration of performance, accessibility, and user experience. The technologies I chose allow for a maintainable and scalable solution.

---

*This is a sample blog post demonstrating the blog functionality. More detailed content will be added soon!*
      `,
      author: "Nickolas Goodis",
      date: "2024-01-15",
      readTime: "8 min read",
      tags: ["React", "Tailwind CSS", "Portfolio", "Web Development"]
    },
    'machine-learning-financial-markets': {
      id: 2,
      title: "Machine Learning in Financial Markets: A Data Science Approach",
      excerpt: "Exploring how machine learning algorithms can be applied to financial market analysis and trading strategy development.",
      content: `
# Machine Learning in Financial Markets

## Coming Soon

This post will cover:
- Data preprocessing for financial time series
- Feature engineering for market data
- Model selection and validation
- Risk management considerations
- Real-world implementation challenges

Stay tuned for the full article!
      `,
      author: "Nickolas Goodis",
      date: "2024-01-10",
      readTime: "12 min read",
      tags: ["Machine Learning", "Finance", "Data Science", "Python"]
    },
    'blockchain-analytics-onchain-data': {
      id: 3,
      title: "Blockchain Analytics: Understanding On-Chain Data",
      excerpt: "Deep dive into blockchain analytics and how to extract meaningful insights from on-chain transaction data.",
      content: `
# Blockchain Analytics: Understanding On-Chain Data

## Coming Soon

This comprehensive guide will explore:
- On-chain data structures
- Analytics tools and platforms
- Transaction pattern analysis
- Address clustering techniques
- DeFi protocol analysis

Full content coming soon!
      `,
      author: "Nickolas Goodis",
      date: "2024-01-05",
      readTime: "10 min read",
      tags: ["Blockchain", "Analytics", "Cryptocurrency", "Data Analysis"]
    }
  };

  const post = blogPosts[slug];

  if (!post) {
    return (
      <PageTransition>
        <div className="min-h-screen bg-gray-50 dark:bg-miami-neutral-900 flex items-center justify-center">
          <Card padding="lg" className="text-center">
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">Post Not Found</h1>
            <p className="text-gray-600 dark:text-gray-300 mb-6">The blog post you're looking for doesn't exist.</p>
            <Link to="/posts">
              <Button variant="primary">
                <ArrowLeft className="w-4 h-4 mr-2" />
                Back to Blog
              </Button>
            </Link>
          </Card>
        </div>
      </PageTransition>
    );
  }

  return (
    <>
      <SEO 
        title={`${post.title} - Nickolas Goodis`}
        description={post.excerpt}
        keywords={post.tags.join(', ')}
        url={`https://nickogoodis.com/posts/${slug}`}
        type="article"
      />
      
      <PageTransition>
        <div className="min-h-screen bg-gray-50 dark:bg-miami-neutral-900 py-12">
          <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
            
            {/* Back to Blog */}
            <AnimatedSection animation="fadeUp" className="mb-8">
              <Link 
                to="/posts" 
                className="inline-flex items-center text-miami-green-600 hover:text-miami-green-700 dark:text-miami-green-400 dark:hover:text-miami-green-300 transition-colors"
              >
                <ArrowLeft className="w-4 h-4 mr-2" />
                Back to Blog
              </Link>
            </AnimatedSection>

            {/* Article Header */}
            <AnimatedSection animation="fadeUp" className="mb-12">
              <Card padding="lg">
                <div className="flex flex-wrap gap-2 mb-4">
                  {post.tags.map((tag, index) => (
                    <Badge key={index} variant="default" size="sm">
                      <Tag className="w-3 h-3 mr-1" />
                      {tag}
                    </Badge>
                  ))}
                </div>
                
                <h1 className="text-3xl sm:text-4xl font-bold text-gray-900 dark:text-white mb-4">
                  {post.title}
                </h1>
                
                <p className="text-xl text-gray-600 dark:text-gray-300 mb-6">
                  {post.excerpt}
                </p>
                
                <div className="flex flex-wrap items-center gap-6 text-sm text-gray-500 dark:text-gray-400">
                  <div className="flex items-center gap-2">
                    <User className="w-4 h-4" />
                    <span>{post.author}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Calendar className="w-4 h-4" />
                    <span>{new Date(post.date).toLocaleDateString()}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Clock className="w-4 h-4" />
                    <span>{post.readTime}</span>
                  </div>
                </div>
              </Card>
            </AnimatedSection>

            {/* Article Content */}
            <AnimatedSection animation="fadeUp" className="mb-12">
              <Card padding="lg">
                <div className="prose prose-lg dark:prose-invert max-w-none">
                  <div className="whitespace-pre-line text-gray-800 dark:text-gray-200 leading-relaxed">
                    {post.content}
                  </div>
                </div>
              </Card>
            </AnimatedSection>

            {/* Share & Navigation */}
            <AnimatedSection animation="fadeUp">
              <Card padding="lg">
                <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
                  <div className="flex items-center gap-4">
                    <span className="text-gray-600 dark:text-gray-300">Share this post:</span>
                    <div className="flex gap-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => {
                          const url = window.location.href;
                          const text = `Check out "${post.title}" by ${post.author}`;
                          if (navigator.share) {
                            navigator.share({ title: post.title, text, url });
                          } else {
                            navigator.clipboard.writeText(url);
                            alert('Link copied to clipboard!');
                          }
                        }}
                      >
                        <Share2 className="w-4 h-4 mr-1" />
                        Share
                      </Button>
                    </div>
                  </div>
                  
                  <Link to="/posts">
                    <Button variant="primary">
                      <ArrowLeft className="w-4 h-4 mr-2" />
                      Back to All Posts
                    </Button>
                  </Link>
                </div>
              </Card>
            </AnimatedSection>

          </div>
        </div>
      </PageTransition>
    </>
  );
};

export default BlogPost;