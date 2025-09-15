import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Send, Mail, User, MessageSquare, CheckCircle, AlertCircle } from 'lucide-react';
import Card from './ui/Card';
import Button from './ui/Button';
import SEO from './SEO';
import PageTransition from './ui/PageTransition';

const ContactForm = () => {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    subject: '',
    message: ''
  });
  const [errors, setErrors] = useState({});
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isSubmitted, setIsSubmitted] = useState(false);

  const validateForm = () => {
    const newErrors = {};

    if (!formData.name.trim()) {
      newErrors.name = 'Name is required';
    }

    if (!formData.email.trim()) {
      newErrors.email = 'Email is required';
    } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(formData.email)) {
      newErrors.email = 'Please enter a valid email address';
    }

    if (!formData.subject.trim()) {
      newErrors.subject = 'Subject is required';
    }

    if (!formData.message.trim()) {
      newErrors.message = 'Message is required';
    } else if (formData.message.length < 10) {
      newErrors.message = 'Message must be at least 10 characters long';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
    
    // Clear error when user starts typing
    if (errors[name]) {
      setErrors(prev => ({
        ...prev,
        [name]: ''
      }));
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!validateForm()) {
      return;
    }

    setIsSubmitting(true);
    
    try {
      // Track form submission
      if (window.gtag) {
        window.gtag('event', 'contact_form_submit', {
          event_category: 'engagement',
          event_label: 'Contact Form'
        });
      }

      // Simulate form submission (replace with actual API call)
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Create mailto link as fallback
      const mailtoLink = `mailto:ncg87@miami.edu?subject=${encodeURIComponent(formData.subject)}&body=${encodeURIComponent(
        `Name: ${formData.name}\nEmail: ${formData.email}\n\nMessage:\n${formData.message}`
      )}`;
      
      window.location.href = mailtoLink;
      setIsSubmitted(true);
      
    } catch (error) {
      console.error('Error submitting form:', error);
    } finally {
      setIsSubmitting(false);
    }
  };

  const inputClasses = "w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-miami-green-500 focus:border-miami-green-500 bg-white dark:bg-miami-neutral-800 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 transition-colors";
  const errorClasses = "text-red-500 text-sm mt-1 flex items-center gap-1";

  if (isSubmitted) {
    return (
      <PageTransition>
        <div className="min-h-screen bg-gray-50 dark:bg-miami-neutral-900 py-12 flex items-center justify-center">
          <Card padding="lg" className="max-w-md mx-auto text-center">
            <CheckCircle className="w-16 h-16 text-green-500 mx-auto mb-4" />
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
              Thank You!
            </h2>
            <p className="text-gray-600 dark:text-gray-300 mb-6">
              Your message has been sent. I'll get back to you as soon as possible.
            </p>
            <Button
              variant="primary"
              onClick={() => {
                setIsSubmitted(false);
                setFormData({ name: '', email: '', subject: '', message: '' });
              }}
            >
              Send Another Message
            </Button>
          </Card>
        </div>
      </PageTransition>
    );
  }

  return (
    <>
      <SEO 
        title="Contact - Nickolas Goodis"
        description="Get in touch with Nickolas Goodis for collaboration opportunities, project discussions, or professional inquiries."
        keywords="contact, collaboration, projects, software engineer, University of Miami"
        url="https://nickogoodis.com/contact"
      />
      
      <PageTransition>
        <div className="min-h-screen bg-gray-50 dark:bg-miami-neutral-900 py-12">
          <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
            
            {/* Header */}
            <motion.div
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
              className="text-center mb-12"
            >
              <h1 className="text-4xl sm:text-5xl font-bold text-gray-900 dark:text-white mb-4">
                Let's Connect
              </h1>
              <p className="text-xl text-gray-600 dark:text-gray-300 max-w-2xl mx-auto">
                Have an interesting project, collaboration opportunity, or just want to chat about technology? 
                I'd love to hear from you!
              </p>
            </motion.div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
              
              {/* Contact Form */}
              <motion.div
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.6, delay: 0.2 }}
              >
                <Card padding="lg">
                  <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center">
                    <MessageSquare className="w-6 h-6 mr-3 text-miami-green-600" />
                    Send a Message
                  </h2>
                  
                  <form onSubmit={handleSubmit} className="space-y-6">
                    <div>
                      <label htmlFor="name" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        Full Name *
                      </label>
                      <div className="relative">
                        <User className="absolute left-3 top-3 w-5 h-5 text-gray-400" />
                        <input
                          type="text"
                          id="name"
                          name="name"
                          value={formData.name}
                          onChange={handleInputChange}
                          className={`${inputClasses} pl-10`}
                          placeholder="Your full name"
                        />
                      </div>
                      {errors.name && (
                        <p className={errorClasses}>
                          <AlertCircle className="w-4 h-4" />
                          {errors.name}
                        </p>
                      )}
                    </div>

                    <div>
                      <label htmlFor="email" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        Email Address *
                      </label>
                      <div className="relative">
                        <Mail className="absolute left-3 top-3 w-5 h-5 text-gray-400" />
                        <input
                          type="email"
                          id="email"
                          name="email"
                          value={formData.email}
                          onChange={handleInputChange}
                          className={`${inputClasses} pl-10`}
                          placeholder="your.email@example.com"
                        />
                      </div>
                      {errors.email && (
                        <p className={errorClasses}>
                          <AlertCircle className="w-4 h-4" />
                          {errors.email}
                        </p>
                      )}
                    </div>

                    <div>
                      <label htmlFor="subject" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        Subject *
                      </label>
                      <input
                        type="text"
                        id="subject"
                        name="subject"
                        value={formData.subject}
                        onChange={handleInputChange}
                        className={inputClasses}
                        placeholder="What's this about?"
                      />
                      {errors.subject && (
                        <p className={errorClasses}>
                          <AlertCircle className="w-4 h-4" />
                          {errors.subject}
                        </p>
                      )}
                    </div>

                    <div>
                      <label htmlFor="message" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        Message *
                      </label>
                      <textarea
                        id="message"
                        name="message"
                        rows="5"
                        value={formData.message}
                        onChange={handleInputChange}
                        className={inputClasses}
                        placeholder="Tell me about your project, idea, or just say hello..."
                      />
                      {errors.message && (
                        <p className={errorClasses}>
                          <AlertCircle className="w-4 h-4" />
                          {errors.message}
                        </p>
                      )}
                    </div>

                    <Button
                      type="submit"
                      variant="primary"
                      size="lg"
                      className="w-full"
                      disabled={isSubmitting}
                    >
                      {isSubmitting ? (
                        <>
                          <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2" />
                          Sending...
                        </>
                      ) : (
                        <>
                          <Send className="w-4 h-4 mr-2" />
                          Send Message
                        </>
                      )}
                    </Button>
                  </form>
                </Card>
              </motion.div>

              {/* Contact Info */}
              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.6, delay: 0.4 }}
                className="space-y-6"
              >
                <Card padding="lg">
                  <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
                    Other Ways to Reach Me
                  </h3>
                  <div className="space-y-4">
                    <div className="flex items-center gap-3">
                      <Mail className="w-5 h-5 text-miami-green-600" />
                      <div>
                        <p className="font-medium text-gray-900 dark:text-white">Email</p>
                        <a 
                          href="mailto:ncg87@miami.edu" 
                          className="text-miami-green-600 hover:text-miami-green-700 transition-colors"
                        >
                          ncg87@miami.edu
                        </a>
                      </div>
                    </div>
                  </div>
                </Card>

                <Card padding="lg">
                  <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
                    What I'm Looking For
                  </h3>
                  <ul className="space-y-3 text-gray-700 dark:text-gray-300">
                    <li className="flex items-start gap-2">
                      <div className="w-2 h-2 bg-miami-green-500 rounded-full mt-2"></div>
                      <span>Internship and full-time opportunities</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <div className="w-2 h-2 bg-miami-green-500 rounded-full mt-2"></div>
                      <span>Open source collaboration projects</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <div className="w-2 h-2 bg-miami-green-500 rounded-full mt-2"></div>
                      <span>Research and academic partnerships</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <div className="w-2 h-2 bg-miami-green-500 rounded-full mt-2"></div>
                      <span>Technical discussions and mentorship</span>
                    </li>
                  </ul>
                </Card>

                <Card padding="lg" className="bg-gradient-to-r from-miami-green-500 to-miami-green-600 text-white">
                  <h3 className="text-xl font-bold mb-3">
                    Response Time
                  </h3>
                  <p className="text-miami-green-100">
                    I typically respond within 24-48 hours. For urgent matters, 
                    please mention it in your subject line.
                  </p>
                </Card>
              </motion.div>

            </div>
          </div>
        </div>
      </PageTransition>
    </>
  );
};

export default ContactForm;