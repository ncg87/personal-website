import React from 'react';
import { motion } from 'framer-motion';
import useInView from '../../hooks/useInView';
import useReducedMotion from '../../hooks/useReducedMotion';

const AnimatedSection = ({ 
  children, 
  animation = 'fadeUp',
  delay = 0,
  duration = 0.6,
  className = '',
  once = true,
  ...props 
}) => {
  const { ref, isInView, hasBeenInView } = useInView();
  const shouldReduceMotion = useReducedMotion();

  const animations = {
    fadeUp: {
      hidden: { opacity: 0, y: shouldReduceMotion ? 0 : 30 },
      visible: { 
        opacity: 1, 
        y: 0,
        transition: {
          duration: shouldReduceMotion ? 0 : duration,
          delay: shouldReduceMotion ? 0 : delay,
          ease: 'easeOut'
        }
      }
    },
    fadeIn: {
      hidden: { opacity: 0 },
      visible: { 
        opacity: 1,
        transition: {
          duration: shouldReduceMotion ? 0 : duration,
          delay: shouldReduceMotion ? 0 : delay,
          ease: 'easeOut'
        }
      }
    },
    slideLeft: {
      hidden: { opacity: 0, x: shouldReduceMotion ? 0 : 50 },
      visible: { 
        opacity: 1, 
        x: 0,
        transition: {
          duration: shouldReduceMotion ? 0 : duration,
          delay: shouldReduceMotion ? 0 : delay,
          ease: 'easeOut'
        }
      }
    },
    slideRight: {
      hidden: { opacity: 0, x: shouldReduceMotion ? 0 : -50 },
      visible: { 
        opacity: 1, 
        x: 0,
        transition: {
          duration: shouldReduceMotion ? 0 : duration,
          delay: shouldReduceMotion ? 0 : delay,
          ease: 'easeOut'
        }
      }
    },
    scale: {
      hidden: { opacity: 0, scale: shouldReduceMotion ? 1 : 0.8 },
      visible: { 
        opacity: 1, 
        scale: 1,
        transition: {
          duration: shouldReduceMotion ? 0 : duration,
          delay: shouldReduceMotion ? 0 : delay,
          ease: 'easeOut'
        }
      }
    },
    stagger: {
      hidden: {},
      visible: {
        transition: {
          staggerChildren: shouldReduceMotion ? 0 : 0.1,
          delayChildren: shouldReduceMotion ? 0 : delay
        }
      }
    }
  };

  const shouldAnimate = once ? hasBeenInView : isInView;

  return (
    <motion.div
      ref={ref}
      variants={animations[animation]}
      initial="hidden"
      animate={shouldAnimate ? "visible" : "hidden"}
      className={className}
      {...props}
    >
      {children}
    </motion.div>
  );
};

export default AnimatedSection;