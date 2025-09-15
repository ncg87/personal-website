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

const ModernProjectsPage = () => {

  const containerVariants = {
    hidden: {},
    visible: {
      transition: {
        staggerChildren: 0.1
      }
    }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { 
      opacity: 1, 
      y: 0,
      transition: {
        duration: 0.5,
        ease: 'easeOut'
      }
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
            <motion.div
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
              className="text-center mb-12"
            >
              <h1 className="text-4xl sm:text-5xl font-bold text-gray-900 dark:text-white mb-4">
                Projects & Work
              </h1>
              <p className="text-xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto">
                A collection of software engineering and data science projects showcasing expertise in 
                blockchain technology, machine learning, and full-stack development.
              </p>
            </motion.div>

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
                  className={project.featured ? 'col-span-full' : ''}
                >
                  <Card 
                    className={`${project.featured ? 'border-2 border-miami-green-200 dark:border-miami-green-800' : ''}`}
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
                      <Link to={`/projects/${project.slug}`}>
                        <Button
                          variant="primary"
                          size="sm"
                        >
                          <Eye className="w-4 h-4 mr-2" />
                          View Case Study
                        </Button>
                      </Link>
                      {project.links.github && (
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => window.open(project.links.github, '_blank')}
                        >
                          <Github className="w-4 h-4 mr-2" />
                          View Code
                        </Button>
                      )}
                      {project.links.website && (
                        <Button
                          variant="secondary"
                          size="sm"
                          onClick={() => window.open(project.links.website, '_blank')}
                        >
                          <ExternalLink className="w-4 h-4 mr-2" />
                          Live Demo
                        </Button>
                      )}
                    </div>
                  </Card>
                </motion.div>
              ))}
            </motion.div>

            {/* Call to Action */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.8, duration: 0.6 }}
              className="text-center mt-16"
            >
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
            </motion.div>
          </div>
        </div>
      </PageTransition>
    </>
  );
};

export default ModernProjectsPage;