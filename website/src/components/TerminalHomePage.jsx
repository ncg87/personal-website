import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Link } from 'react-router-dom';
import { ArrowRight, Download, Eye, Code, Brain, Award, Terminal, Play } from 'lucide-react';
import { getFeaturedProjects } from '../data/projects';
import Card from './ui/Card';
import Button from './ui/Button';
import Badge from './ui/Badge';
import SEO from './SEO';
import OptimizedImage from './ui/OptimizedImage';
import useReducedMotion from '../hooks/useReducedMotion';

const TerminalHomePage = () => {
  const [bootSequence, setBootSequence] = useState(0);
  const [terminalLines, setTerminalLines] = useState([]);
  const [showContent, setShowContent] = useState(false);
  const [skipAnimation, setSkipAnimation] = useState(false);
  const shouldReduceMotion = useReducedMotion();
  const featuredProjects = getFeaturedProjects().slice(0, 3);

  const terminalCommands = [
    { text: "$ initializing portfolio system...", delay: 0 },
    { text: "$ loading user profile: nickolas_goodis", delay: 200 },
    { text: "$ mounting projects directory...", delay: 400 },
    { text: "$ scanning achievements: [✓] 3.9 GPA [✓] 12+ Projects [✓] Research", delay: 600 },
    { text: "$ connecting to university: miami.edu", delay: 800 },
    { text: "$ loading skills: python, react, machine_learning, blockchain", delay: 1000 },
    { text: "$ portfolio system ready ✓", delay: 1200 },
    { text: "$ welcome to nickolas goodis dev environment", delay: 1400 }
  ];

  useEffect(() => {
    // Reset state when component mounts
    setBootSequence(0);
    setTerminalLines([]);
    setShowContent(false);
    setSkipAnimation(false);

    if (shouldReduceMotion) {
      setShowContent(true);
      return;
    }

    const timer = setTimeout(() => {
      setBootSequence(1);
    }, 100);

    const timeouts = [];

    terminalCommands.forEach((cmd, index) => {
      const timeout = setTimeout(() => {
        setTerminalLines(prev => [...prev, cmd.text]);
        if (index === terminalCommands.length - 1) {
          const finalTimeout = setTimeout(() => setShowContent(true), 300);
          timeouts.push(finalTimeout);
        }
      }, cmd.delay + 300);
      timeouts.push(timeout);
    });

    return () => {
      clearTimeout(timer);
      timeouts.forEach(timeout => clearTimeout(timeout));
    };
  }, [shouldReduceMotion]); // Removed skipAnimation dependency to prevent reset loop

  // Keyboard support for skipping animation
  useEffect(() => {
    const handleKeyPress = (e) => {
      if ((e.key === 'Escape' || e.key === ' ') && !showContent) {
        handleSkipAnimation();
      }
    };

    document.addEventListener('keydown', handleKeyPress);
    return () => document.removeEventListener('keydown', handleKeyPress);
  }, [showContent]);

  const handleSkipAnimation = () => {
    setSkipAnimation(true);
    setShowContent(true);
    setBootSequence(1);
  };

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

  // Terminal boot sequence animation
  if (!shouldReduceMotion && !skipAnimation && (!showContent || bootSequence < 1)) {
    return (
      <div className="min-h-screen bg-black flex items-center justify-center overflow-hidden relative">
        <SEO 
          title="Nickolas Goodis - Software Engineer & Data Scientist"
          description="Software engineer and data scientist from University of Miami. Specializing in React, TypeScript, Python, machine learning, and blockchain development."
          keywords="software engineer, data scientist, React, TypeScript, Python, machine learning, blockchain, University of Miami, portfolio"
          url="https://nickogoodis.com"
          type="website"
        />
        
        {/* Skip Animation Button */}
        <motion.button
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1 }}
          onClick={handleSkipAnimation}
          className="absolute top-4 right-4 text-miami-green-400 hover:text-white text-sm font-mono bg-gray-900/50 px-3 py-2 rounded border border-miami-green-500/30 hover:border-miami-green-500 transition-colors z-10"
        >
          [ESC] Skip Animation
        </motion.button>
        
        <div className="w-full max-w-4xl mx-auto p-8">
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5 }}
            className="bg-gray-900 rounded-lg border border-gray-700 shadow-2xl overflow-hidden"
          >
            {/* Terminal Header */}
            <div className="bg-gray-800 px-4 py-3 flex items-center gap-2 border-b border-gray-700">
              <div className="flex gap-2">
                <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                <div className="w-3 h-3 bg-green-500 rounded-full"></div>
              </div>
              <div className="flex items-center gap-2 ml-4">
                <Terminal className="w-4 h-4 text-gray-400" />
                <span className="text-sm text-gray-400 font-mono">portfolio.system</span>
              </div>
            </div>

            {/* Terminal Content */}
            <div className="p-6 h-96 overflow-hidden">
              <div className="font-mono text-green-400 space-y-2">
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.2 }}
                  className="text-gray-500"
                >
                  Nickolas Goodis Portfolio System v2.0
                </motion.div>
                
                <AnimatePresence>
                  {terminalLines.map((line, index) => (
                    <motion.div
                      key={index}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ duration: 0.3 }}
                      className="text-green-400"
                    >
                      {line}
                    </motion.div>
                  ))}
                </AnimatePresence>

                {terminalLines.length > 0 && (
                  <motion.div
                    animate={{ opacity: [1, 0, 1] }}
                    transition={{ duration: 1, repeat: Infinity }}
                    className="inline-block w-2 h-5 bg-green-400 ml-1"
                  />
                )}
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    );
  }

  // Main content with swooping animations
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
          
          {/* Animated Background Grid */}
          <div className="absolute inset-0 overflow-hidden">
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 0.1 }}
              transition={{ duration: 2, delay: shouldReduceMotion ? 0 : 1 }}
              className="absolute inset-0"
              style={{
                backgroundImage: `radial-gradient(circle at 1px 1px, rgba(5, 80, 48, 0.15) 1px, transparent 0)`,
                backgroundSize: '50px 50px'
              }}
            />
            
            {/* Floating Elements */}
            <motion.div
              initial={{ opacity: 0, scale: 0, rotate: 45 }}
              animate={{ 
                opacity: 0.1, 
                scale: 1, 
                rotate: shouldReduceMotion ? 45 : 405 
              }}
              transition={{ 
                duration: shouldReduceMotion ? 0 : 3, 
                delay: shouldReduceMotion ? 0 : 1,
                ease: "easeOut" 
              }}
              className="absolute top-20 right-20 w-64 h-64 border-2 border-miami-green-500 rounded-lg"
            />
            
            <motion.div
              initial={{ opacity: 0, scale: 0, rotate: -45 }}
              animate={{ 
                opacity: 0.05, 
                scale: 1, 
                rotate: shouldReduceMotion ? -45 : -405 
              }}
              transition={{ 
                duration: shouldReduceMotion ? 0 : 3, 
                delay: shouldReduceMotion ? 0 : 1.5,
                ease: "easeOut" 
              }}
              className="absolute bottom-20 left-20 w-96 h-96 border-2 border-miami-orange-500 rounded-full"
            />
          </div>

          <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
              
              {/* Hero Content */}
              <div className="text-center lg:text-left">
                
                {/* Status Badge */}
                <motion.div
                  initial={{ opacity: 0, y: shouldReduceMotion ? 0 : -30, scale: shouldReduceMotion ? 1 : 0.5 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  transition={{ 
                    duration: shouldReduceMotion ? 0 : 0.6, 
                    delay: shouldReduceMotion ? 0 : 0.2,
                    type: "spring",
                    damping: 20
                  }}
                >
                  <Badge variant="primary" size="md" className="mb-6 inline-flex items-center gap-2">
                    <motion.div
                      animate={{ scale: [1, 1.2, 1] }}
                      transition={{ duration: 2, repeat: Infinity }}
                      className="w-2 h-2 bg-green-400 rounded-full"
                    />
                    System Online - Available for Opportunities
                  </Badge>
                </motion.div>

                {/* Main Title */}
                <motion.div
                  initial={{ opacity: 0, x: shouldReduceMotion ? 0 : -100 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ 
                    duration: shouldReduceMotion ? 0 : 0.8, 
                    delay: shouldReduceMotion ? 0 : 0.4,
                    type: "spring",
                    damping: 25
                  }}
                >
                  <h1 className="text-5xl sm:text-6xl lg:text-7xl font-bold text-gray-900 dark:text-white mb-6">
                    <motion.span 
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ delay: shouldReduceMotion ? 0 : 0.6, duration: shouldReduceMotion ? 0 : 0.6 }}
                      className="bg-gradient-to-r from-miami-green-600 to-miami-orange-500 bg-clip-text text-transparent block"
                    >
                      ./nickolas
                    </motion.span>
                    <motion.span
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ delay: shouldReduceMotion ? 0 : 0.8, duration: shouldReduceMotion ? 0 : 0.6 }}
                      className="block"
                    >
                      --dev-mode
                    </motion.span>
                  </h1>
                </motion.div>

                {/* Description */}
                <motion.div
                  initial={{ opacity: 0, x: shouldReduceMotion ? 0 : -50 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ 
                    duration: shouldReduceMotion ? 0 : 0.8, 
                    delay: shouldReduceMotion ? 0 : 1.0,
                    ease: "easeOut"
                  }}
                >
                  <p className="text-xl sm:text-2xl text-gray-600 dark:text-gray-300 mb-8 leading-relaxed font-mono">
                    <span className="text-miami-green-600">$</span> software_engineer --specialization=
                    <span className="text-miami-green-600 font-semibold">AI</span> 
                    <span className="text-miami-orange-500 font-semibold"> blockchain</span> 
                    <span className="text-miami-green-600 font-semibold"> ML</span>
                    <br />
                    <span className="text-miami-green-600">$</span> university="Miami" gpa=3.9 status=active
                  </p>
                </motion.div>

                {/* Action Buttons */}
                <motion.div 
                  initial={{ opacity: 0, y: shouldReduceMotion ? 0 : 50 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ 
                    duration: shouldReduceMotion ? 0 : 0.6, 
                    delay: shouldReduceMotion ? 0 : 1.2,
                    type: "spring",
                    damping: 20
                  }}
                  className="flex flex-col sm:flex-row gap-4 justify-center lg:justify-start mb-12"
                >
                  <Link to="/projects">
                    <Button variant="primary" size="lg" className="group font-mono">
                      <Play className="w-5 h-5 mr-2" />
                      ./execute_projects
                      <ArrowRight className="w-4 h-4 ml-2 group-hover:translate-x-1 transition-transform" />
                    </Button>
                  </Link>
                  <Button 
                    variant="outline" 
                    size="lg"
                    onClick={handleDownloadResume}
                    className="font-mono"
                  >
                    <Download className="w-5 h-5 mr-2" />
                    cat resume.pdf
                  </Button>
                </motion.div>

                {/* Stats Terminal */}
                <motion.div 
                  initial={{ opacity: 0, scale: shouldReduceMotion ? 1 : 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ 
                    duration: shouldReduceMotion ? 0 : 0.6, 
                    delay: shouldReduceMotion ? 0 : 1.4,
                    type: "spring",
                    damping: 20
                  }}
                  className="bg-black/80 backdrop-blur-sm rounded-lg p-6 border border-miami-green-500/30"
                >
                  <div className="grid grid-cols-3 gap-6 font-mono text-sm">
                    {stats.map((stat, index) => (
                      <motion.div
                        key={index}
                        initial={{ opacity: 0, x: shouldReduceMotion ? 0 : -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ 
                          delay: shouldReduceMotion ? 0 : 1.6 + (index * 0.1),
                          duration: shouldReduceMotion ? 0 : 0.4
                        }}
                        className="text-center"
                      >
                        <div className="text-miami-green-400 mb-2 flex justify-center">
                          {stat.icon}
                        </div>
                        <div className="text-2xl font-bold text-white">
                          {stat.value}
                        </div>
                        <div className="text-miami-green-400/80 text-xs">
                          {stat.label}
                        </div>
                      </motion.div>
                    ))}
                  </div>
                </motion.div>
              </div>

              {/* Hero Image with Terminal Effect */}
              <motion.div
                initial={{ 
                  opacity: 0, 
                  x: shouldReduceMotion ? 0 : 100, 
                  rotateY: shouldReduceMotion ? 0 : -45 
                }}
                animate={{ opacity: 1, x: 0, rotateY: 0 }}
                transition={{ 
                  duration: shouldReduceMotion ? 0 : 0.8, 
                  delay: shouldReduceMotion ? 0 : 0.6,
                  type: "spring",
                  damping: 25
                }}
                className="relative perspective-1000"
              >
                <div className="relative z-10">
                  <motion.div
                    whileHover={shouldReduceMotion ? {} : { 
                      scale: 1.02,
                      rotateY: 5,
                      transition: { duration: 0.3 }
                    }}
                    className="relative"
                  >
                    <OptimizedImage
                      src="/me.jpg"
                      alt="Nickolas Goodis"
                      className="w-full max-w-md mx-auto rounded-2xl shadow-2xl border-4 border-miami-green-500/30"
                      priority={true}
                    />
                    
                    {/* Hologram Effect Overlay */}
                    <motion.div
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 0.1 }}
                      transition={{ delay: shouldReduceMotion ? 0 : 1, duration: 0.6 }}
                      className="absolute inset-0 bg-gradient-to-t from-miami-green-500/20 via-transparent to-miami-orange-500/20 rounded-2xl"
                    />
                    
                    {/* Scanning Lines Effect */}
                    <motion.div
                      initial={{ y: '100%', opacity: 0 }}
                      animate={{ y: '-100%', opacity: [0, 0.8, 0] }}
                      transition={{
                        duration: shouldReduceMotion ? 0 : 1.5,
                        delay: shouldReduceMotion ? 0 : 1.5,
                        repeat: shouldReduceMotion ? 0 : Infinity,
                        repeatDelay: 4
                      }}
                      className="absolute inset-0 bg-gradient-to-b from-transparent via-miami-green-400/30 to-transparent h-8 rounded-2xl"
                    />
                  </motion.div>
                </div>
                
                {/* Glowing Background */}
                <motion.div
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 0.3, scale: 1.1 }}
                  transition={{ delay: shouldReduceMotion ? 0 : 1.2, duration: 0.6 }}
                  className="absolute inset-0 bg-gradient-to-tr from-miami-green-500 to-miami-orange-500 rounded-2xl blur-xl"
                />
              </motion.div>

            </div>
          </div>

          {/* Animated Scroll Indicator */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: shouldReduceMotion ? 0 : 1.8, duration: 0.6 }}
            className="absolute bottom-8 left-1/2 transform -translate-x-1/2"
          >
            <motion.div
              animate={shouldReduceMotion ? {} : {
                y: [0, 10, 0],
                transition: { duration: 2, repeat: Infinity }
              }}
              className="flex flex-col items-center gap-2 text-gray-400"
            >
              <span className="text-sm font-mono">scroll_down()</span>
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
          </motion.div>
        </section>

        {/* Featured Projects Section */}
        <section className="py-20 bg-white dark:bg-miami-neutral-900">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            
            <motion.div
              initial={{ opacity: 0, y: shouldReduceMotion ? 0 : 50 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-100px" }}
              transition={{ duration: shouldReduceMotion ? 0 : 0.8 }}
              className="text-center mb-16"
            >
              <h2 className="text-4xl sm:text-5xl font-bold text-gray-900 dark:text-white mb-6 font-mono">
                <span className="text-miami-green-600">$</span> ls featured_projects/
              </h2>
              <p className="text-xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto">
                A curated collection of my most impactful work in software engineering, machine learning, and blockchain technology.
              </p>
            </motion.div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
              {featuredProjects.map((project, index) => (
                <motion.div
                  key={project.id}
                  initial={{ 
                    opacity: 0, 
                    y: shouldReduceMotion ? 0 : 50,
                    rotateX: shouldReduceMotion ? 0 : 15
                  }}
                  whileInView={{ opacity: 1, y: 0, rotateX: 0 }}
                  viewport={{ once: true, margin: "-50px" }}
                  transition={{ 
                    duration: shouldReduceMotion ? 0 : 0.8, 
                    delay: shouldReduceMotion ? 0 : index * 0.2,
                    type: "spring",
                    damping: 25
                  }}
                  whileHover={shouldReduceMotion ? {} : { 
                    y: -10,
                    rotateX: 5,
                    transition: { duration: 0.3 }
                  }}
                >
                  <Card className="h-full group cursor-pointer border-2 border-transparent hover:border-miami-green-500/30 transition-all duration-300" padding="lg">
                    <div className="mb-4">
                      <div className="flex items-center gap-2 mb-3">
                        <Badge variant="primary" size="xs">{project.category}</Badge>
                        <Badge variant="secondary" size="xs">Featured</Badge>
                      </div>
                      <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-2 group-hover:text-miami-green-600 transition-colors font-mono">
                        ./{project.title.toLowerCase().replace(/\s+/g, '_')}
                      </h3>
                      <p className="text-gray-600 dark:text-gray-300 text-sm leading-relaxed">
                        {project.overview.challenge}
                      </p>
                    </div>

                    <div className="flex flex-wrap gap-1 mb-4">
                      {project.technologies.slice(0, 3).map((tech, idx) => (
                        <Badge key={idx} variant="default" size="xs" className="font-mono">
                          {tech}
                        </Badge>
                      ))}
                      {project.technologies.length > 3 && (
                        <Badge variant="default" size="xs" className="font-mono">
                          +{project.technologies.length - 3}
                        </Badge>
                      )}
                    </div>

                    <Link to={`/projects/${project.slug}`}>
                      <Button variant="ghost" size="sm" className="w-full group-hover:bg-miami-green-50 dark:group-hover:bg-miami-green-900/20 font-mono">
                        ./execute --view
                        <ArrowRight className="w-4 h-4 ml-2 group-hover:translate-x-1 transition-transform" />
                      </Button>
                    </Link>
                  </Card>
                </motion.div>
              ))}
            </div>

            <motion.div
              initial={{ opacity: 0 }}
              whileInView={{ opacity: 1 }}
              viewport={{ once: true }}
              transition={{ delay: shouldReduceMotion ? 0 : 0.5, duration: shouldReduceMotion ? 0 : 0.8 }}
              className="text-center mt-12"
            >
              <Link to="/projects">
                <Button variant="outline" size="lg" className="font-mono">
                  cd /projects && ls -la
                  <ArrowRight className="w-4 h-4 ml-2" />
                </Button>
              </Link>
            </motion.div>

          </div>
        </section>

        {/* Terminal CTA Section */}
        <section className="py-20 bg-gradient-to-r from-miami-green-500 to-miami-green-600 relative overflow-hidden">
          
          {/* Background Terminal Pattern */}
          <div className="absolute inset-0 opacity-10">
            <div className="absolute inset-0 font-mono text-xs leading-relaxed text-white/20 whitespace-pre overflow-hidden">
              {Array.from({ length: 20 }, (_, i) => (
                <div key={i} className="animate-pulse" style={{ animationDelay: `${i * 0.1}s` }}>
                  {`$ ./portfolio --mode=collaboration --status=active\n`}
                </div>
              ))}
            </div>
          </div>

          <div className="relative max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
            
            <motion.div
              initial={{ opacity: 0, scale: shouldReduceMotion ? 1 : 0.8 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
              transition={{ duration: shouldReduceMotion ? 0 : 0.8 }}
              className="text-white"
            >
              <h2 className="text-4xl sm:text-5xl font-bold mb-6 font-mono">
                $ init_collaboration()
              </h2>
              <p className="text-xl text-miami-green-100 mb-8 max-w-2xl mx-auto">
                Ready to build something innovative? Let's connect and create impactful solutions together.
              </p>
              <div className="flex flex-col sm:flex-row gap-4 justify-center">
                <Link to="/contact">
                  <Button 
                    variant="outline" 
                    size="lg"
                    className="bg-white text-miami-green-600 border-white hover:bg-gray-50 font-mono"
                  >
                    ./contact --action=collaborate
                    <ArrowRight className="w-4 h-4 ml-2" />
                  </Button>
                </Link>
                <Link to="/about">
                  <Button 
                    variant="outline" 
                    size="lg"
                    className="bg-transparent text-white border-white hover:bg-white hover:text-miami-green-600 font-mono"
                  >
                    man nickolas_goodis
                  </Button>
                </Link>
              </div>
            </motion.div>

          </div>
        </section>

      </div>
    </>
  );
};

export default TerminalHomePage;