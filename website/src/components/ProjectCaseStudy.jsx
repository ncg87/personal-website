import React from 'react';
import { useParams, Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { ArrowLeft, Github, ExternalLink, Calendar, Users, Award, Target, Lightbulb, Zap } from 'lucide-react';
import { getProjectBySlug } from '../data/projects';
import Card from './ui/Card';
import Badge from './ui/Badge';
import Button from './ui/Button';
import SEO from './SEO';
import PageTransition from './ui/PageTransition';

const ProjectCaseStudy = () => {
  const { slug } = useParams();
  const project = getProjectBySlug(slug);

  if (!project) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-miami-neutral-900 flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">Project Not Found</h1>
          <Link to="/projects" className="text-miami-green-600 hover:text-miami-green-700">
            ← Back to Projects
          </Link>
        </div>
      </div>
    );
  }

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
        duration: 0.6,
        ease: 'easeOut'
      }
    }
  };

  return (
    <>
      <SEO 
        title={`${project.title} - Project Case Study`}
        description={project.overview.challenge}
        keywords={`${project.technologies.join(', ')}, case study, project, software engineering`}
        url={`https://nickogoodis.com/projects/${project.slug}`}
      />
      
      <PageTransition>
        <div className="min-h-screen bg-gray-50 dark:bg-miami-neutral-900 py-12">
          <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
            
            {/* Back Button */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5 }}
              className="mb-8"
            >
              <Link 
                to="/projects" 
                className="inline-flex items-center text-miami-green-600 hover:text-miami-green-700 transition-colors"
              >
                <ArrowLeft className="w-4 h-4 mr-2" />
                Back to Projects
              </Link>
            </motion.div>

            <motion.div
              variants={containerVariants}
              initial="hidden"
              animate="visible"
              className="space-y-8"
            >
              
              {/* Header */}
              <motion.div variants={itemVariants}>
                <Card padding="lg">
                  <div className="flex flex-col lg:flex-row lg:items-center justify-between mb-6">
                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-3">
                        <Badge variant="primary" size="sm">{project.category}</Badge>
                        <Badge variant={project.status === 'Live' || project.status === 'Deployed' ? 'success' : 'default'} size="sm">
                          {project.status}
                        </Badge>
                        {project.featured && (
                          <Badge variant="secondary" size="sm">Featured</Badge>
                        )}
                      </div>
                      <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-2">
                        {project.title}
                      </h1>
                      <p className="text-xl text-miami-green-600 dark:text-miami-green-400 mb-4">
                        {project.subtitle}
                      </p>
                    </div>
                  </div>

                  {/* Project Meta */}
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                    <div className="flex items-center gap-2 text-gray-600 dark:text-gray-300">
                      <Calendar className="w-5 h-5" />
                      <span><strong>Timeline:</strong> {project.timeline}</span>
                    </div>
                    <div className="flex items-center gap-2 text-gray-600 dark:text-gray-300">
                      <Users className="w-5 h-5" />
                      <span><strong>Team:</strong> {project.team}</span>
                    </div>
                    <div className="flex items-center gap-2 text-gray-600 dark:text-gray-300">
                      <Award className="w-5 h-5" />
                      <span><strong>Role:</strong> {project.role}</span>
                    </div>
                  </div>

                  {/* Action Buttons */}
                  <div className="flex flex-wrap gap-3">
                    {project.links.github && (
                      <Button
                        variant="outline"
                        onClick={() => window.open(project.links.github, '_blank')}
                      >
                        <Github className="w-4 h-4 mr-2" />
                        View Code
                      </Button>
                    )}
                    {project.links.website && (
                      <Button
                        variant="primary"
                        onClick={() => window.open(project.links.website, '_blank')}
                      >
                        <ExternalLink className="w-4 h-4 mr-2" />
                        Live Demo
                      </Button>
                    )}
                  </div>
                </Card>
              </motion.div>

              {/* Overview */}
              <motion.div variants={itemVariants}>
                <Card padding="lg">
                  <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center">
                    <Target className="w-6 h-6 mr-3 text-miami-green-600" />
                    Project Overview
                  </h2>
                  
                  <div className="space-y-6">
                    <div>
                      <h3 className="font-semibold text-lg text-gray-900 dark:text-white mb-2">The Challenge</h3>
                      <p className="text-gray-700 dark:text-gray-300">{project.overview.challenge}</p>
                    </div>
                    
                    <div>
                      <h3 className="font-semibold text-lg text-gray-900 dark:text-white mb-2">The Solution</h3>
                      <p className="text-gray-700 dark:text-gray-300">{project.overview.solution}</p>
                    </div>
                    
                    <div>
                      <h3 className="font-semibold text-lg text-gray-900 dark:text-white mb-2">The Impact</h3>
                      <p className="text-gray-700 dark:text-gray-300">{project.overview.impact}</p>
                    </div>
                  </div>
                </Card>
              </motion.div>

              {/* Key Metrics */}
              <motion.div variants={itemVariants}>
                <Card padding="lg">
                  <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">Key Metrics & Results</h2>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {Object.entries(project.metrics).map(([key, value]) => (
                      <div key={key} className="text-center p-4 bg-gray-50 dark:bg-miami-neutral-800 rounded-lg">
                        <div className="text-2xl font-bold text-miami-green-600 dark:text-miami-green-400 mb-1">
                          {value}
                        </div>
                        <div className="text-sm text-gray-600 dark:text-gray-400">
                          {key}
                        </div>
                      </div>
                    ))}
                  </div>
                </Card>
              </motion.div>

              {/* Technologies */}
              <motion.div variants={itemVariants}>
                <Card padding="lg">
                  <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">Technologies Used</h2>
                  <div className="flex flex-wrap gap-2">
                    {project.technologies.map((tech, index) => (
                      <Badge key={index} variant="default" size="md">
                        {tech}
                      </Badge>
                    ))}
                  </div>
                </Card>
              </motion.div>

              {/* Key Features */}
              <motion.div variants={itemVariants}>
                <Card padding="lg">
                  <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center">
                    <Zap className="w-6 h-6 mr-3 text-miami-green-600" />
                    Key Features
                  </h2>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {project.keyFeatures.map((feature, index) => (
                      <div key={index} className="flex items-start gap-3">
                        <div className="w-2 h-2 bg-miami-green-500 rounded-full mt-2 flex-shrink-0"></div>
                        <span className="text-gray-700 dark:text-gray-300">{feature}</span>
                      </div>
                    ))}
                  </div>
                </Card>
              </motion.div>

              {/* Technical Challenges */}
              {project.technicalChallenges && (
                <motion.div variants={itemVariants}>
                  <Card padding="lg">
                    <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center">
                      <Lightbulb className="w-6 h-6 mr-3 text-miami-green-600" />
                      Technical Challenges & Solutions
                    </h2>
                    <div className="space-y-6">
                      {project.technicalChallenges.map((item, index) => (
                        <div key={index} className="border-l-4 border-miami-green-500 pl-6">
                          <h3 className="font-semibold text-lg text-gray-900 dark:text-white mb-2">
                            Challenge: {item.challenge}
                          </h3>
                          <p className="text-gray-700 dark:text-gray-300 mb-2">
                            <strong>Solution:</strong> {item.solution}
                          </p>
                          <p className="text-miami-green-600 dark:text-miami-green-400">
                            <strong>Result:</strong> {item.result}
                          </p>
                        </div>
                      ))}
                    </div>
                  </Card>
                </motion.div>
              )}

              {/* Call to Action */}
              <motion.div variants={itemVariants}>
                <Card padding="lg" className="text-center">
                  <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
                    Interested in Similar Projects?
                  </h3>
                  <p className="text-gray-600 dark:text-gray-300 mb-6">
                    Let's discuss how we can build something amazing together.
                  </p>
                  <div className="flex flex-col sm:flex-row gap-4 justify-center">
                    <Button
                      variant="primary"
                      onClick={() => window.location.href = '/projects'}
                    >
                      View More Projects
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

            </motion.div>
          </div>
        </div>
      </PageTransition>
    </>
  );
};

export default ProjectCaseStudy;