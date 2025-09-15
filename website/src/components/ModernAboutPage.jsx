import React from 'react';
import { MapPin, GraduationCap, Code, Brain, Award, Coffee, Music, Camera, Gamepad2 } from 'lucide-react';
import Card from './ui/Card';
import Badge from './ui/Badge';
import Button from './ui/Button';
import SEO from './SEO';
import PageTransition from './ui/PageTransition';
import OptimizedImage from './ui/OptimizedImage';
import AnimatedSection from './ui/AnimatedSection';
import useReducedMotion from '../hooks/useReducedMotion';

const ModernAboutPage = () => {
  const shouldReduceMotion = useReducedMotion();

  const skills = {
    'Programming Languages': ['Python', 'Java', 'C/C++', 'JavaScript', 'TypeScript', 'Rust', 'Solidity'],
    'Frameworks & Libraries': ['React', 'Node.js', 'PyTorch', 'Flask', 'Next.js', 'Tailwind CSS'],
    'Data & ML': ['Machine Learning', 'Deep Learning', 'Data Analysis', 'Statistics', 'Computer Vision', 'NLP'],
    'Cloud & Tools': ['AWS', 'Docker', 'Git', 'PostgreSQL', 'MongoDB', 'Neo4j']
  };

  const experiences = [
    {
      title: 'Nuclear Data Science Intern',
      company: 'NextEra Energy',
      period: 'Aug 2025 - Present',
      location: 'Juno Beach, FL',
      description: 'Building MCP servers and ReAct AI agents for nuclear compliance monitoring.',
      icon: <Code className="w-5 h-5" />
    },
    {
      title: 'Undergraduate Researcher',
      company: 'University of Miami',
      period: 'Jan 2024 - Present',
      location: 'Coral Gables, FL',
      description: 'Advanced ML research under Director of Graduate Studies, focusing on N-shot learning and medical image segmentation.',
      icon: <Brain className="w-5 h-5" />
    },
    {
      title: 'Quantitative Trader Intern',
      company: 'Greenland Risk Management',
      period: 'Aug 2024 - Dec 2024',
      location: 'Dallas, TX',
      description: 'Developed news-based futures trading algorithms with LLM integration.',
      icon: <Award className="w-5 h-5" />
    }
  ];

  const interests = [
    { name: 'Coffee Enthusiast', icon: <Coffee className="w-5 h-5" />, description: 'Always exploring new brewing methods' },
    { name: 'Music Production', icon: <Music className="w-5 h-5" />, description: 'Creating beats and soundscapes' },
    { name: 'Photography', icon: <Camera className="w-5 h-5" />, description: 'Capturing moments and landscapes' },
    { name: 'Gaming', icon: <Gamepad2 className="w-5 h-5" />, description: 'Strategy games and competitive esports' },
  ];

  const education = {
    university: 'University of Miami',
    location: 'Coral Gables, Florida',
    period: 'August 2022 - May 2026',
    gpa: '3.9',
    sat: '1520',
    degrees: [
      'Bachelor of Science in Computer Science',
      'Bachelor of Science in Mathematics',
      'Bachelor of Science in Data Science and Artificial Intelligence'
    ]
  };

  return (
    <>
      <SEO 
        title="About - Nickolas Goodis"
        description="Learn about Nickolas Goodis, a Computer Science, Mathematics, and Data Science student at University of Miami with expertise in machine learning and software development."
        keywords="about, University of Miami, computer science, mathematics, data science, machine learning, software engineer"
        url="https://nickogoodis.com/about"
      />
      
      <PageTransition>
        <div className="min-h-screen bg-gray-50 dark:bg-miami-neutral-900 py-12">
          <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
            
            <div className="space-y-12">
              
              {/* Hero Section */}
              <AnimatedSection animation="fadeUp">
                <Card padding="lg" className="overflow-hidden">
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-center">
                    <div>
                      <h1 className="text-4xl sm:text-5xl font-bold text-gray-900 dark:text-white mb-4">
                        Hey, I'm Nicko! 👋
                      </h1>
                      <p className="text-xl text-miami-green-600 dark:text-miami-green-400 mb-6">
                        Software Engineer & Data Scientist
                      </p>
                      <p className="text-gray-700 dark:text-gray-300 text-lg leading-relaxed mb-6">
                        I'm a passionate triple-major student at the University of Miami, diving deep into 
                        Computer Science, Mathematics, and Data Science & AI. I love building things that make 
                        a difference – from blockchain analytics platforms to AI research tools.
                      </p>
                      <div className="flex items-center gap-4 text-sm text-gray-600 dark:text-gray-400">
                        <div className="flex items-center gap-2">
                          <MapPin className="w-4 h-4" />
                          <span>Miami, FL</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <GraduationCap className="w-4 h-4" />
                          <span>Class of 2026</span>
                        </div>
                      </div>
                    </div>
                    <div className="relative">
                      <OptimizedImage
                        src="/me.jpg"
                        alt="Nickolas Goodis"
                        className="w-full max-w-md mx-auto rounded-2xl shadow-lg"
                        priority={true}
                      />
                    </div>
                  </div>
                </Card>
              </AnimatedSection>

              {/* Education */}
              <AnimatedSection animation="fadeUp">
                <Card padding="lg">
                  <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-6 flex items-center">
                    <GraduationCap className="w-8 h-8 mr-3 text-miami-green-600" />
                    Education
                  </h2>
                  <div className="bg-gradient-to-r from-miami-green-50 to-miami-orange-50 dark:from-miami-green-900/20 dark:to-miami-orange-900/20 rounded-xl p-6">
                    <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
                      {education.university}
                    </h3>
                    <p className="text-miami-green-600 dark:text-miami-green-400 mb-4">
                      {education.location} • {education.period}
                    </p>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                      <div>
                        <h4 className="font-semibold text-gray-900 dark:text-white mb-3">Triple Major</h4>
                        <ul className="space-y-2">
                          {education.degrees.map((degree, index) => (
                            <li key={index} className="text-gray-700 dark:text-gray-300 flex items-start gap-2">
                              <div className="w-2 h-2 bg-miami-green-500 rounded-full mt-2 flex-shrink-0"></div>
                              {degree}
                            </li>
                          ))}
                        </ul>
                      </div>
                      <div>
                        <h4 className="font-semibold text-gray-900 dark:text-white mb-3">Academic Performance</h4>
                        <div className="space-y-2">
                          <div className="flex justify-between">
                            <span className="text-gray-700 dark:text-gray-300">GPA:</span>
                            <span className="font-semibold text-miami-green-600">{education.gpa}/4.0</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-700 dark:text-gray-300">SAT:</span>
                            <span className="font-semibold text-miami-green-600">{education.sat}/1600</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </Card>
              </AnimatedSection>

              {/* Experience */}
              <AnimatedSection animation="fadeUp">
                <Card padding="lg">
                  <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-6">
                    Professional Experience
                  </h2>
                  <div className="space-y-6">
                    {experiences.map((exp, index) => (
                      <div key={index} className="border-l-4 border-miami-green-500 pl-6 relative">
                        <div className="absolute left-[-8px] top-0 w-4 h-4 bg-miami-green-500 rounded-full"></div>
                        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-2">
                          <h3 className="text-xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
                            {exp.icon}
                            {exp.title}
                          </h3>
                          <span className="text-sm text-gray-600 dark:text-gray-400">{exp.period}</span>
                        </div>
                        <p className="text-miami-green-600 dark:text-miami-green-400 font-medium mb-1">
                          {exp.company} • {exp.location}
                        </p>
                        <p className="text-gray-700 dark:text-gray-300">
                          {exp.description}
                        </p>
                      </div>
                    ))}
                  </div>
                </Card>
              </AnimatedSection>

              {/* Skills */}
              <AnimatedSection animation="fadeUp">
                <Card padding="lg">
                  <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-6 flex items-center">
                    <Code className="w-8 h-8 mr-3 text-miami-green-600" />
                    Technical Skills
                  </h2>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                    {Object.entries(skills).map(([category, skillList]) => (
                      <div key={category}>
                        <h3 className="font-semibold text-lg text-gray-900 dark:text-white mb-3">
                          {category}
                        </h3>
                        <div className="flex flex-wrap gap-2">
                          {skillList.map((skill, index) => (
                            <Badge key={index} variant="default" size="sm">
                              {skill}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </Card>
              </AnimatedSection>

              {/* Interests & Hobbies */}
              <AnimatedSection animation="fadeUp">
                <Card padding="lg">
                  <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-6">
                    Beyond Code
                  </h2>
                  <p className="text-gray-700 dark:text-gray-300 mb-6">
                    When I'm not coding or studying, you'll find me exploring these passions:
                  </p>
                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
                    {interests.map((interest, index) => (
                      <div key={index} className="text-center p-4 bg-gray-50 dark:bg-miami-neutral-800 rounded-xl">
                        <div className="flex justify-center mb-3 text-miami-green-600">
                          {interest.icon}
                        </div>
                        <h3 className="font-semibold text-gray-900 dark:text-white mb-2">
                          {interest.name}
                        </h3>
                        <p className="text-sm text-gray-600 dark:text-gray-400">
                          {interest.description}
                        </p>
                      </div>
                    ))}
                  </div>
                </Card>
              </AnimatedSection>

              {/* Call to Action */}
              <AnimatedSection animation="fadeUp">
                <Card padding="lg" className="text-center bg-gradient-to-r from-miami-green-500 to-miami-green-600 text-white">
                  <h3 className="text-2xl font-bold mb-4">
                    Let's Build Something Amazing Together
                  </h3>
                  <p className="text-miami-green-100 mb-6 max-w-2xl mx-auto">
                    I'm always excited to collaborate on innovative projects, discuss research opportunities, 
                    or explore new technologies. Feel free to reach out!
                  </p>
                  <div className="flex flex-col sm:flex-row gap-4 justify-center">
                    <Button
                      variant="outline"
                      className="bg-white text-miami-green-600 border-white hover:bg-gray-50"
                      onClick={() => window.location.href = '/projects'}
                    >
                      View My Work
                    </Button>
                    <Button
                      variant="outline"
                      className="bg-transparent text-white border-white hover:bg-white hover:text-miami-green-600"
                      onClick={() => window.location.href = 'mailto:ncg87@miami.edu'}
                    >
                      Get In Touch
                    </Button>
                  </div>
                </Card>
              </AnimatedSection>

            </div>
          </div>
        </div>
      </PageTransition>
    </>
  );
};

export default ModernAboutPage;