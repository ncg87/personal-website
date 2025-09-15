import React from 'react';
import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import { ArrowRight, Download, Eye, Code, Brain, Award } from 'lucide-react';
import { getFeaturedProjects } from '../data/projects';
import Card from './ui/Card';
import Button from './ui/Button';
import Badge from './ui/Badge';
import SEO from './SEO';
import AnimatedSection from './ui/AnimatedSection';
import OptimizedImage from './ui/OptimizedImage';
import useReducedMotion from '../hooks/useReducedMotion';

const EnhancedHomePage = () => {
  const shouldReduceMotion = useReducedMotion();
  const featuredProjects = getFeaturedProjects().slice(0, 3);

  const handleDownloadResume = () => {
    if (window.gtag) {
      window.gtag('event', 'resume_download_hero', {
        event_category: 'engagement',
        event_label: 'Hero Section'
      });
    }
    
    const link = document.createElement('a');
    link.href = '/resume.pdf';
    link.download = 'Nickolas_Goodis_Resume.pdf';
    link.click();
  };

  const stats = [
    { icon: <Code className="w-8 h-8" />, value: '12+', label: 'Projects Built' },
    { icon: <Brain className="w-8 h-8" />, value: '3.9', label: 'GPA' },
    { icon: <Award className="w-8 h-8" />, value: '600%', label: 'Trading Returns' }
  ];

  const heroVariants = {
    hidden: {},
    visible: {
      transition: {
        staggerChildren: shouldReduceMotion ? 0 : 0.2,
        delayChildren: shouldReduceMotion ? 0 : 0.3
      }
    }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: shouldReduceMotion ? 0 : 30 },
    visible: { 
      opacity: 1, 
      y: 0,
      transition: {
        duration: shouldReduceMotion ? 0 : 0.8,
        ease: 'easeOut'
      }
    }
  };

  const floatingVariants = {
    animate: shouldReduceMotion ? {} : {
      y: [-5, 5, -5],
      transition: {
        duration: 3,
        repeat: Infinity,
        ease: "easeInOut"
      }
    }
  };

  return (
    <>
      <SEO 
        title="Nickolas Goodis - Software Engineer & Data Scientist"
        description="Software engineer and data scientist from University of Miami. Specializing in React, TypeScript, Python, machine learning, and blockchain development."
        keywords="software engineer, data scientist, React, TypeScript, Python, machine learning, blockchain, University of Miami, portfolio"
        url="https://nickogoodis.com"
        type="website"
      />
      
      <div className="min-h-screen bg-gradient-to-br from-gray-50 via-white to-gray-100 dark:from-miami-neutral-900 dark:via-miami-neutral-900 dark:to-gray-900">
        {/* Hero Section */}
        <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
          {/* Background Elements */}
          <div className="absolute inset-0 overflow-hidden">
            <motion.div
              animate={shouldReduceMotion ? {} : {
                rotate: 360,
                transition: { duration: 50, repeat: Infinity, ease: "linear" }
              }}
              className="absolute -top-40 -right-40 w-80 h-80 bg-miami-green-500/10 rounded-full blur-3xl"
            />
            <motion.div
              animate={shouldReduceMotion ? {} : {
                rotate: -360,
                transition: { duration: 40, repeat: Infinity, ease: "linear" }
              }}
              className="absolute -bottom-40 -left-40 w-96 h-96 bg-miami-orange-500/10 rounded-full blur-3xl"
            />
          </div>

          <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
              
              {/* Hero Content */}
              <motion.div
                variants={heroVariants}
                initial="hidden"
                animate="visible"
                className="text-center lg:text-left"
              >
                <motion.div variants={itemVariants}>
                  <Badge variant="primary" size="md" className="mb-6">
                    Available for Opportunities
                  </Badge>
                </motion.div>

                <motion.h1 
                  variants={itemVariants}
                  className="text-5xl sm:text-6xl lg:text-7xl font-bold text-gray-900 dark:text-white mb-6"
                >
                  <span className="bg-gradient-to-r from-miami-green-600 to-miami-orange-500 bg-clip-text text-transparent">
                    Nickolas
                  </span>
                  <br />
                  <span>Goodis</span>
                </motion.h1>

                <motion.p 
                  variants={itemVariants}
                  className="text-xl sm:text-2xl text-gray-600 dark:text-gray-300 mb-8 leading-relaxed"
                >
                  Software Engineer & Data Scientist building the future with{' '}
                  <span className="text-miami-green-600 font-semibold">AI</span>,{' '}
                  <span className="text-miami-orange-500 font-semibold">Blockchain</span>, and{' '}
                  <span className="text-miami-green-600 font-semibold">Machine Learning</span>
                </motion.p>

                <motion.div 
                  variants={itemVariants}
                  className="flex flex-col sm:flex-row gap-4 justify-center lg:justify-start mb-12"
                >
                  <Link to="/projects">
                    <Button variant="primary" size="lg" className="group">
                      <Eye className="w-5 h-5 mr-2" />
                      View My Work
                      <ArrowRight className="w-4 h-4 ml-2 group-hover:translate-x-1 transition-transform" />
                    </Button>
                  </Link>
                  <Button 
                    variant="outline" 
                    size="lg"
                    onClick={handleDownloadResume}
                  >
                    <Download className="w-5 h-5 mr-2" />
                    Download Resume
                  </Button>
                </motion.div>

                {/* Stats */}
                <motion.div 
                  variants={itemVariants}
                  className="grid grid-cols-3 gap-8"
                >
                  {stats.map((stat, index) => (
                    <motion.div
                      key={index}
                      whileHover={shouldReduceMotion ? {} : { scale: 1.05 }}
                      className="text-center"
                    >
                      <div className="text-miami-green-600 dark:text-miami-green-400 mb-2 flex justify-center">
                        {stat.icon}
                      </div>
                      <div className="text-2xl font-bold text-gray-900 dark:text-white">
                        {stat.value}
                      </div>
                      <div className="text-sm text-gray-600 dark:text-gray-400">
                        {stat.label}
                      </div>
                    </motion.div>
                  ))}
                </motion.div>
              </motion.div>

              {/* Hero Image */}
              <motion.div
                variants={floatingVariants}
                animate="animate"
                className="relative"
              >
                <div className="relative z-10">
                  <OptimizedImage
                    src="/me.jpg"
                    alt="Nickolas Goodis"
                    className="w-full max-w-md mx-auto rounded-2xl shadow-2xl"
                    priority={true}
                  />
                </div>
                <div className="absolute inset-0 bg-gradient-to-tr from-miami-green-500 to-miami-orange-500 rounded-2xl blur-xl opacity-20 scale-105" />
              </motion.div>

            </div>
          </div>

          {/* Scroll Indicator */}
          <motion.div
            animate={shouldReduceMotion ? {} : {
              y: [0, 10, 0],
              transition: { duration: 2, repeat: Infinity }
            }}
            className="absolute bottom-8 left-1/2 transform -translate-x-1/2"
          >
            <div className="w-6 h-10 border-2 border-gray-400 rounded-full flex justify-center">
              <motion.div
                animate={shouldReduceMotion ? {} : {
                  y: [0, 12, 0],
                  transition: { duration: 2, repeat: Infinity }
                }}
                className="w-1 h-3 bg-gray-400 rounded-full mt-2"
              />
            </div>
          </motion.div>
        </section>

        {/* Featured Projects Section */}
        <section className="py-20 bg-white dark:bg-miami-neutral-900">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            
            <AnimatedSection animation="fadeUp" className="text-center mb-16">
              <h2 className="text-4xl sm:text-5xl font-bold text-gray-900 dark:text-white mb-6">
                Featured Projects
              </h2>
              <p className="text-xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto">
                A showcase of my most impactful work in software engineering, machine learning, and blockchain technology.
              </p>
            </AnimatedSection>

            <AnimatedSection animation="stagger" className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
              {featuredProjects.map((project, index) => (
                <motion.div
                  key={project.id}
                  variants={{
                    hidden: { opacity: 0, y: shouldReduceMotion ? 0 : 30 },
                    visible: { 
                      opacity: 1, 
                      y: 0,
                      transition: {
                        duration: shouldReduceMotion ? 0 : 0.6,
                        ease: 'easeOut'
                      }
                    }
                  }}
                  whileHover={shouldReduceMotion ? {} : { 
                    y: -5,
                    transition: { duration: 0.2 }
                  }}
                >
                  <Card className="h-full group cursor-pointer" padding="lg">
                    <div className="mb-4">
                      <div className="flex items-center gap-2 mb-3">
                        <Badge variant="primary" size="xs">{project.category}</Badge>
                        <Badge variant="secondary" size="xs">Featured</Badge>
                      </div>
                      <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-2 group-hover:text-miami-green-600 transition-colors">
                        {project.title}
                      </h3>
                      <p className="text-gray-600 dark:text-gray-300 text-sm leading-relaxed">
                        {project.overview.challenge}
                      </p>
                    </div>

                    <div className="flex flex-wrap gap-1 mb-4">
                      {project.technologies.slice(0, 3).map((tech, idx) => (
                        <Badge key={idx} variant="default" size="xs">
                          {tech}
                        </Badge>
                      ))}
                      {project.technologies.length > 3 && (
                        <Badge variant="default" size="xs">
                          +{project.technologies.length - 3} more
                        </Badge>
                      )}
                    </div>

                    <Link to={`/projects/${project.slug}`}>
                      <Button variant="ghost" size="sm" className="w-full group-hover:bg-miami-green-50 dark:group-hover:bg-miami-green-900/20">
                        View Case Study
                        <ArrowRight className="w-4 h-4 ml-2 group-hover:translate-x-1 transition-transform" />
                      </Button>
                    </Link>
                  </Card>
                </motion.div>
              ))}
            </AnimatedSection>

            <AnimatedSection animation="fadeUp" delay={0.5} className="text-center mt-12">
              <Link to="/projects">
                <Button variant="outline" size="lg">
                  View All Projects
                  <ArrowRight className="w-4 h-4 ml-2" />
                </Button>
              </Link>
            </AnimatedSection>

          </div>
        </section>

        {/* Call to Action Section */}
        <section className="py-20 bg-gradient-to-r from-miami-green-500 to-miami-green-600">
          <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
            
            <AnimatedSection animation="fadeUp" className="text-white">
              <h2 className="text-4xl sm:text-5xl font-bold mb-6">
                Let's Build Something Amazing
              </h2>
              <p className="text-xl text-miami-green-100 mb-8 max-w-2xl mx-auto">
                I'm always excited to collaborate on innovative projects and explore new technologies. 
                Let's connect and discuss how we can create something impactful together.
              </p>
              <div className="flex flex-col sm:flex-row gap-4 justify-center">
                <Link to="/contact">
                  <Button 
                    variant="outline" 
                    size="lg"
                    className="bg-white text-miami-green-600 border-white hover:bg-gray-50"
                  >
                    Get In Touch
                    <ArrowRight className="w-4 h-4 ml-2" />
                  </Button>
                </Link>
                <Link to="/about">
                  <Button 
                    variant="outline" 
                    size="lg"
                    className="bg-transparent text-white border-white hover:bg-white hover:text-miami-green-600"
                  >
                    Learn More About Me
                  </Button>
                </Link>
              </div>
            </AnimatedSection>

          </div>
        </section>

      </div>
    </>
  );
};

export default EnhancedHomePage;