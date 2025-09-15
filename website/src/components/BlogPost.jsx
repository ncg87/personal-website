import React from 'react';
import { useParams, Link, Navigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { ArrowLeft, Calendar, Clock, User, Tag, Share2, ChevronRight } from 'lucide-react';
import Card from './ui/Card';
import Badge from './ui/Badge';
import Button from './ui/Button';
import SEO from './SEO';
import PageTransition from './ui/PageTransition';
import AnimatedSection from './ui/AnimatedSection';
import { getPostBySlug, getPostsByTag } from '../data/blogPosts';

const BlogPost = () => {
  const { slug } = useParams();
  const post = getPostBySlug(slug);

  if (!post) {
    return <Navigate to="/posts" replace />;
  }

  const sharePost = () => {
    if (navigator.share) {
      navigator.share({
        title: post.title,
        text: post.excerpt,
        url: window.location.href,
      });
    } else {
      navigator.clipboard.writeText(window.location.href);
      // Could add a toast notification here
    }
  };

  // Simple markdown-like processing for basic formatting
  const processContent = (content) => {
    return content
      .split('\n')
      .map((line, index) => {
        // Handle headers
        if (line.startsWith('# ')) {
          return <h1 key={index} className="text-3xl font-bold text-gray-900 dark:text-white mt-8 mb-4">{line.slice(2)}</h1>;
        }
        if (line.startsWith('## ')) {
          return <h2 key={index} className="text-2xl font-bold text-gray-900 dark:text-white mt-6 mb-3">{line.slice(3)}</h2>;
        }
        if (line.startsWith('### ')) {
          return <h3 key={index} className="text-xl font-bold text-gray-900 dark:text-white mt-5 mb-2">{line.slice(4)}</h3>;
        }
        
        // Handle code blocks
        if (line.startsWith('```')) {
          return null; // We'll handle these in a more sophisticated way later
        }
        
        // Handle empty lines
        if (line.trim() === '') {
          return <br key={index} />;
        }
        
        // Handle bullet points
        if (line.startsWith('- ')) {
          return (
            <li key={index} className="text-gray-700 dark:text-gray-300 ml-4 mb-2">
              {line.slice(2)}
            </li>
          );
        }
        
        // Handle bold text and code spans
        const processInlineFormatting = (text) => {
          // Bold text
          text = text.replace(/\*\*(.*?)\*\*/g, '<strong class="font-semibold">$1</strong>');
          // Code spans
          text = text.replace(/`(.*?)`/g, '<code class="bg-gray-100 dark:bg-gray-800 px-1 py-0.5 rounded text-sm font-mono">$1</code>');
          return text;
        };
        
        // Regular paragraph
        return (
          <p 
            key={index} 
            className="text-gray-700 dark:text-gray-300 leading-relaxed mb-4"
            dangerouslySetInnerHTML={{ __html: processInlineFormatting(line) }}
          />
        );
      })
      .filter(Boolean);
  };

  return (
    <>
      <SEO 
        title={`${post.title} - Nickolas Goodis`}
        description={post.excerpt}
        keywords={post.tags.join(', ')}
        url={`https://nickogoodis.com/posts/${post.slug}`}
        type="article"
        article={{
          publishedTime: post.date,
          author: post.author,
          tags: post.tags
        }}
      />
      
      <PageTransition>
        <div className="min-h-screen bg-gray-50 dark:bg-miami-neutral-900 py-12">
          <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
            
            {/* Back Button */}
            <AnimatedSection animation="fadeUp" className="mb-8">
              <Link to="/posts">
                <Button variant="ghost" size="sm" className="mb-4">
                  <ArrowLeft className="w-4 h-4 mr-2" />
                  Back to Blog
                </Button>
              </Link>
            </AnimatedSection>

            {/* Article Header */}
            <AnimatedSection animation="fadeUp" delay={0.1}>
              <Card className="mb-8" padding="lg">
                <div className="flex flex-wrap items-center gap-2 mb-4">
                  {post.featured && (
                    <Badge variant="primary" size="sm">Featured</Badge>
                  )}
                  {post.tags.slice(0, 3).map((tag, idx) => (
                    <Badge key={idx} variant="default" size="sm">
                      {tag}
                    </Badge>
                  ))}
                </div>
                
                <h1 className="text-3xl sm:text-4xl lg:text-5xl font-bold text-gray-900 dark:text-white mb-4 leading-tight">
                  {post.title}
                </h1>
                
                <p className="text-xl text-gray-600 dark:text-gray-300 mb-6 leading-relaxed">
                  {post.excerpt}
                </p>
                
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                  <div className="flex items-center gap-6 text-sm text-gray-500 dark:text-gray-400">
                    <div className="flex items-center gap-2">
                      <User className="w-4 h-4" />
                      <span>{post.author}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Calendar className="w-4 h-4" />
                      <span>{new Date(post.date).toLocaleDateString('en-US', {
                        year: 'numeric',
                        month: 'long',
                        day: 'numeric'
                      })}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Clock className="w-4 h-4" />
                      <span>{post.readTime}</span>
                    </div>
                  </div>
                  
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={sharePost}
                    className="self-start"
                  >
                    <Share2 className="w-4 h-4 mr-2" />
                    Share
                  </Button>
                </div>
              </Card>
            </AnimatedSection>

            {/* Article Content */}
            <AnimatedSection animation="fadeUp" delay={0.2}>
              <Card className="mb-8" padding="lg">
                <article className="prose prose-lg dark:prose-invert max-w-none">
                  <div className="space-y-4">
                    {processContent(post.content)}
                  </div>
                </article>
              </Card>
            </AnimatedSection>

            {/* Author Bio */}
            <AnimatedSection animation="fadeUp" delay={0.3}>
              <Card className="mb-8" padding="lg">
                <div className="flex items-center gap-4">
                  <div className="w-16 h-16 bg-miami-green-500 rounded-full flex items-center justify-center text-white text-xl font-bold">
                    {post.author.split(' ').map(n => n[0]).join('')}
                  </div>
                  <div>
                    <h3 className="text-lg font-bold text-gray-900 dark:text-white">
                      {post.author}
                    </h3>
                    <p className="text-gray-600 dark:text-gray-300">
                      Software Engineer & Data Scientist passionate about building innovative solutions 
                      with modern technologies and machine learning.
                    </p>
                  </div>
                </div>
              </Card>
            </AnimatedSection>

            {/* Navigation */}
            <AnimatedSection animation="fadeUp" delay={0.4}>
              <div className="flex flex-col sm:flex-row gap-4 justify-between">
                <Link to="/posts">
                  <Button variant="outline" className="w-full sm:w-auto">
                    <ArrowLeft className="w-4 h-4 mr-2" />
                    All Posts
                  </Button>
                </Link>
                
                <div className="flex flex-col sm:flex-row gap-4">
                  <Link to="/projects">
                    <Button variant="primary" className="w-full sm:w-auto">
                      View Projects
                      <ChevronRight className="w-4 h-4 ml-2" />
                    </Button>
                  </Link>
                  <Link to="/contact">
                    <Button variant="ghost" className="w-full sm:w-auto">
                      Get In Touch
                    </Button>
                  </Link>
                </div>
              </div>
            </AnimatedSection>

          </div>
        </div>
      </PageTransition>
    </>
  );
};

export default BlogPost;