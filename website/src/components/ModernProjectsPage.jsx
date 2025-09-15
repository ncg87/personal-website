import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { ExternalLink, Github, Calendar, Tag, Eye } from 'lucide-react';
import { projects } from '../data/projects';
import Card from './ui/Card';
import Badge from './ui/Badge';
import Button from './ui/Button';
import SEO from './SEO';
import PageTransition from './ui/PageTransition';
import AnimatedSection from './ui/AnimatedSection';
import useReducedMotion from '../hooks/useReducedMotion';

const ModernProjectsPage = () => {
  const shouldReduceMotion = useReducedMotion();

  const containerVariants = {
    hidden: {},
    visible: {
      transition: {
        staggerChildren: shouldReduceMotion ? 0 : 0.15
      }
    }
  };

  const itemVariants = {
    hidden: { 
      opacity: 0, 
      y: shouldReduceMotion ? 0 : 30,
      scale: shouldReduceMotion ? 1 : 0.96
    },
    visible: { 
      opacity: 1, 
      y: 0,
      scale: 1,
      transition: {
        duration: shouldReduceMotion ? 0 : 0.7,
        ease: [0.25, 0.25, 0, 1]
      }
    }
  };

  const cardHoverVariants = {
    hover: shouldReduceMotion ? {} : {
      y: -8,
      scale: 1.02,
      boxShadow: "0 20px 25px -5px rgba(0, 80, 48, 0.1), 0 10px 10px -5px rgba(0, 80, 48, 0.04)",
      transition: {
        duration: 0.3,
        ease: "easeOut"
      }
    },
    tap: shouldReduceMotion ? {} : {
      scale: 0.98,
      transition: { duration: 0.1 }
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'Live':
      case 'Deployed':
        return 'success';
      case 'In Development':
        return 'warning';
      case 'Completed':
      case 'Open Source':
        return 'primary';
      default:
        return 'default';
    }
  };

  return (
    <>
      <SEO 
        title="Projects - Nickolas Goodis"
        description="Portfolio of software engineering and data science projects including blockchain analytics, machine learning algorithms, and full-stack applications."
        keywords="projects, portfolio, blockchain, machine learning, React, Python, Rust, software engineering"
        url="https://nickogoodis.com/projects"
      />
      
      <PageTransition>
        <div className="min-h-screen bg-gray-50 dark:bg-miami-neutral-900 py-12">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            {/* Header */}
            <AnimatedSection animation="fadeUp" className="text-center mb-12">
              <h1 className="text-4xl sm:text-5xl font-bold text-gray-900 dark:text-white mb-4">
                Projects & Work
              </h1>
              <p className="text-xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto">
                A collection of software engineering and data science projects showcasing expertise in 
                blockchain technology, machine learning, and full-stack development.
              </p>
            </AnimatedSection>

            {/* Featured Projects */}
            <motion.div
              variants={containerVariants}
              initial="hidden"
              animate="visible"
              className="space-y-8"
            >
              {projects.map((project, index) => (
                <motion.div
                  key={project.id}
                  variants={itemVariants}
                  whileHover="hover"
                  whileTap="tap"
                  className={`group cursor-pointer ${project.featured ? 'col-span-full' : ''}`}
                >
                  <motion.div variants={cardHoverVariants}>
                    <Card 
                      className={`overflow-hidden transition-all duration-300 ${project.featured ? 'border-2 border-miami-green-200 dark:border-miami-green-800' : ''} group-hover:border-miami-green-300 dark:group-hover:border-miami-green-700`}
                      padding="lg"
                    >
                    <div className="flex flex-col lg:flex-row lg:items-center justify-between mb-4">
                      <div className="flex-1">
                        <div className="flex items-center gap-3 mb-2">
                          <h3 className="text-2xl font-bold text-gray-900 dark:text-white">
                            {project.title}
                          </h3>
                          <Badge variant="default" size="xs">
                            {project.category}
                          </Badge>
                          <Badge variant={getStatusColor(project.status)} size="sm">
                            {project.status}
                          </Badge>
                          {project.featured && (
                            <Badge variant="secondary" size="sm">
                              Featured
                            </Badge>
                          )}
                        </div>
                        <p className="text-miami-green-600 dark:text-miami-green-400 text-lg mb-2">
                          {project.subtitle}
                        </p>
                        <p className="text-gray-700 dark:text-gray-300 mb-4 leading-relaxed">
                          {project.overview.challenge}
                        </p>
                      </div>
                    </div>

                    {/* Technologies */}
                    <div className="mb-4">
                      <div className="flex flex-wrap gap-2">
                        {project.technologies.map((tech, idx) => (
                          <Badge key={idx} variant="default" size="sm">
                            <Tag className="w-3 h-3 mr-1" />
                            {tech}
                          </Badge>
                        ))}
                      </div>
                    </div>

                    {/* Metrics */}
                    {project.metrics && (
                      <div className="mb-6 p-4 bg-gray-50 dark:bg-miami-neutral-800 rounded-lg">
                        <h4 className="font-semibold text-gray-900 dark:text-white mb-2">Key Metrics</h4>
                        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-sm">
                          {Object.entries(project.metrics).map(([key, value]) => (
                            <div key={key} className="text-center">
                              <div className="font-medium text-miami-green-600 dark:text-miami-green-400">
                                {value}
                              </div>
                              <div className="text-gray-600 dark:text-gray-400 capitalize">
                                {key.replace(/([A-Z])/g, ' $1').trim()}
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Actions */}
                    <div className="flex flex-wrap gap-3">
                      <motion.div
                        whileHover={shouldReduceMotion ? {} : { scale: 1.05 }}
                        whileTap={shouldReduceMotion ? {} : { scale: 0.95 }}
                      >
                        <Link to={`/projects/${project.slug}`}>
                          <Button
                            variant="primary"
                            size="sm"
                            className="group"
                          >
                            <Eye className="w-4 h-4 mr-2 group-hover:scale-110 transition-transform" />
                            View Case Study
                          </Button>
                        </Link>
                      </motion.div>
                      {project.links.github && (
                        <motion.div
                          whileHover={shouldReduceMotion ? {} : { scale: 1.05 }}
                          whileTap={shouldReduceMotion ? {} : { scale: 0.95 }}
                        >
                          <Button
                            variant="outline"
                            size="sm"
                            className="group"
                            onClick={() => window.open(project.links.github, '_blank')}
                          >
                            <Github className="w-4 h-4 mr-2 group-hover:rotate-12 transition-transform" />
                            View Code
                          </Button>
                        </motion.div>
                      )}
                      {project.links.website && (
                        <motion.div
                          whileHover={shouldReduceMotion ? {} : { scale: 1.05 }}
                          whileTap={shouldReduceMotion ? {} : { scale: 0.95 }}
                        >
                          <Button
                            variant="secondary"
                            size="sm"
                            className="group"
                            onClick={() => window.open(project.links.website, '_blank')}
                          >
                            <ExternalLink className="w-4 h-4 mr-2 group-hover:translate-x-1 group-hover:-translate-y-1 transition-transform" />
                            Live Demo
                          </Button>
                        </motion.div>
                      )}
                      </div>
                    </Card>
                  </motion.div>
                </motion.div>
              ))}
            </motion.div>

            {/* Call to Action */}
            <AnimatedSection animation="fadeUp" delay={0.5} className="text-center mt-16">
              <Card padding="lg" className="max-w-2xl mx-auto">
                <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
                  Interested in Collaboration?
                </h3>
                <p className="text-gray-600 dark:text-gray-300 mb-6">
                  I'm always excited to work on innovative projects and explore new technologies. 
                  Let's discuss how we can build something amazing together.
                </p>
                <div className="flex flex-col sm:flex-row gap-4 justify-center">
                  <Button
                    variant="primary"
                    onClick={() => window.location.href = '/resume'}
                  >
                    View Resume
                  </Button>
                  <Button
                    variant="outline"
                    onClick={() => window.location.href = 'mailto:ncg87@miami.edu'}
                  >
                    Contact Me
                  </Button>
                </div>
              </Card>
            </AnimatedSection>
          </div>
        </div>
      </PageTransition>
    </>
  );
};

export default ModernProjectsPage;