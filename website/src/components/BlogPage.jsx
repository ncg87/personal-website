import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Calendar, Clock, ArrowRight, User, Tag } from 'lucide-react';
import Card from './ui/Card';
import Badge from './ui/Badge';
import Button from './ui/Button';
import SEO from './SEO';
import PageTransition from './ui/PageTransition';
import AnimatedSection from './ui/AnimatedSection';

const BlogPage = () => {
  // Sample blog posts data - in a real app, this would come from a CMS or MDX files
  const blogPosts = [
    {
      id: 1,
      title: "Building a Modern Portfolio with React and Tailwind CSS",
      excerpt: "Learn how I built this portfolio using React, Tailwind CSS, and Framer Motion with a focus on performance and accessibility.",
      content: "Coming soon...",
      author: "Nickolas Goodis",
      date: "2024-01-15",
      readTime: "8 min read",
      tags: ["React", "Tailwind CSS", "Portfolio", "Web Development"],
      slug: "building-modern-portfolio-react-tailwind",
      featured: true
    },
    {
      id: 2,
      title: "Machine Learning in Financial Markets: A Data Science Approach",
      excerpt: "Exploring how machine learning algorithms can be applied to financial market analysis and trading strategy development.",
      content: "Coming soon...",
      author: "Nickolas Goodis",
      date: "2024-01-10",
      readTime: "12 min read",
      tags: ["Machine Learning", "Finance", "Data Science", "Python"],
      slug: "machine-learning-financial-markets",
      featured: true
    },
    {
      id: 3,
      title: "Blockchain Analytics: Understanding On-Chain Data",
      excerpt: "Deep dive into blockchain analytics and how to extract meaningful insights from on-chain transaction data.",
      content: "Coming soon...",
      author: "Nickolas Goodis",
      date: "2024-01-05",
      readTime: "10 min read",
      tags: ["Blockchain", "Analytics", "Cryptocurrency", "Data Analysis"],
      slug: "blockchain-analytics-onchain-data",
      featured: false
    }
  ];

  const featuredPosts = blogPosts.filter(post => post.featured);
  const regularPosts = blogPosts.filter(post => !post.featured);

  return (
    <>
      <SEO 
        title="Blog - Nickolas Goodis"
        description="Technical blog posts about software engineering, machine learning, blockchain technology, and data science by Nickolas Goodis."
        keywords="blog, technical writing, software engineering, machine learning, blockchain, data science, programming tutorials"
        url="https://nickogoodis.com/posts"
      />
      
      <PageTransition>
        <div className="min-h-screen bg-gray-50 dark:bg-miami-neutral-900 py-12">
          <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
            
            {/* Header */}
            <AnimatedSection animation="fadeUp" className="text-center mb-12">
              <h1 className="text-4xl sm:text-5xl font-bold text-gray-900 dark:text-white mb-4">
                Blog & Insights
              </h1>
              <p className="text-xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto">
                Sharing thoughts and experiences on software engineering, machine learning, 
                blockchain technology, and the latest in computer science.
              </p>
            </AnimatedSection>

            {/* Featured Posts */}
            {featuredPosts.length > 0 && (
              <section className="mb-16">
                <AnimatedSection animation="fadeUp" className="mb-8">
                  <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">Featured Posts</h2>
                </AnimatedSection>
                
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                  {featuredPosts.map((post, index) => (
                    <AnimatedSection 
                      key={post.id}
                      animation="fadeUp"
                      delay={index * 0.1}
                    >
                      <Card className="h-full group cursor-pointer hover:shadow-lg transition-all duration-300" padding="lg">
                        <div className="flex items-center gap-2 mb-3">
                          <Badge variant="primary" size="xs">Featured</Badge>
                          {post.tags.slice(0, 2).map((tag, idx) => (
                            <Badge key={idx} variant="default" size="xs">
                              {tag}
                            </Badge>
                          ))}
                        </div>
                        
                        <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-3 group-hover:text-miami-green-600 transition-colors">
                          {post.title}
                        </h3>
                        
                        <p className="text-gray-600 dark:text-gray-300 mb-4 leading-relaxed">
                          {post.excerpt}
                        </p>
                        
                        <div className="flex items-center justify-between text-sm text-gray-500 dark:text-gray-400 mb-4">
                          <div className="flex items-center gap-4">
                            <div className="flex items-center gap-1">
                              <User className="w-4 h-4" />
                              <span>{post.author}</span>
                            </div>
                            <div className="flex items-center gap-1">
                              <Calendar className="w-4 h-4" />
                              <span>{new Date(post.date).toLocaleDateString()}</span>
                            </div>
                            <div className="flex items-center gap-1">
                              <Clock className="w-4 h-4" />
                              <span>{post.readTime}</span>
                            </div>
                          </div>
                        </div>
                        
                        <Link to={`/posts/${post.slug}`}>
                          <Button variant="ghost" size="sm" className="w-full group-hover:bg-miami-green-50 dark:group-hover:bg-miami-green-900/20">
                            Read More
                            <ArrowRight className="w-4 h-4 ml-2 group-hover:translate-x-1 transition-transform" />
                          </Button>
                        </Link>
                      </Card>
                    </AnimatedSection>
                  ))}
                </div>
              </section>
            )}

            {/* Regular Posts */}
            {regularPosts.length > 0 && (
              <section>
                <AnimatedSection animation="fadeUp" className="mb-8">
                  <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">All Posts</h2>
                </AnimatedSection>
                
                <div className="space-y-6">
                  {regularPosts.map((post, index) => (
                    <AnimatedSection 
                      key={post.id}
                      animation="fadeUp"
                      delay={index * 0.1}
                    >
                      <Card className="group cursor-pointer hover:shadow-md transition-all duration-300" padding="lg">
                        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                          <div className="flex-1">
                            <div className="flex items-center gap-2 mb-2">
                              {post.tags.slice(0, 3).map((tag, idx) => (
                                <Badge key={idx} variant="default" size="xs">
                                  {tag}
                                </Badge>
                              ))}
                            </div>
                            
                            <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-2 group-hover:text-miami-green-600 transition-colors">
                              {post.title}
                            </h3>
                            
                            <p className="text-gray-600 dark:text-gray-300 mb-3">
                              {post.excerpt}
                            </p>
                            
                            <div className="flex items-center gap-4 text-sm text-gray-500 dark:text-gray-400">
                              <div className="flex items-center gap-1">
                                <Calendar className="w-3 h-3" />
                                <span>{new Date(post.date).toLocaleDateString()}</span>
                              </div>
                              <div className="flex items-center gap-1">
                                <Clock className="w-3 h-3" />
                                <span>{post.readTime}</span>
                              </div>
                            </div>
                          </div>
                          
                          <div className="md:ml-6">
                            <Link to={`/posts/${post.slug}`}>
                              <Button variant="outline" size="sm">
                                Read More
                                <ArrowRight className="w-4 h-4 ml-2" />
                              </Button>
                            </Link>
                          </div>
                        </div>
                      </Card>
                    </AnimatedSection>
                  ))}
                </div>
              </section>
            )}

            {/* Coming Soon Message */}
            <AnimatedSection animation="fadeUp" className="text-center mt-16">
              <Card padding="lg" className="max-w-2xl mx-auto">
                <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
                  More Content Coming Soon
                </h3>
                <p className="text-gray-600 dark:text-gray-300 mb-6">
                  I'm working on more technical articles and project deep-dives. 
                  Follow my work or get in touch if you have specific topics you'd like me to cover.
                </p>
                <div className="flex flex-col sm:flex-row gap-4 justify-center">
                  <Link to="/projects">
                    <Button variant="primary">
                      View My Projects
                    </Button>
                  </Link>
                  <Link to="/contact">
                    <Button variant="outline">
                      Get In Touch
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

export default BlogPage;