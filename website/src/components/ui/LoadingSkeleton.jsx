import React from 'react';
import { motion } from 'framer-motion';
import useReducedMotion from '../../hooks/useReducedMotion';

const LoadingSkeleton = ({ 
  width = 'w-full', 
  height = 'h-4', 
  className = '', 
  count = 1,
  variant = 'default' 
}) => {
  const shouldReduceMotion = useReducedMotion();

  const shimmerVariants = {
    initial: { x: '-100%' },
    animate: {
      x: '100%',
      transition: {
        repeat: Infinity,
        duration: shouldReduceMotion ? 0 : 1.5,
        ease: 'easeInOut'
      }
    }
  };

  const pulseVariants = {
    initial: { opacity: 0.6 },
    animate: {
      opacity: [0.6, 1, 0.6],
      transition: {
        repeat: Infinity,
        duration: shouldReduceMotion ? 0 : 2,
        ease: 'easeInOut'
      }
    }
  };

  const getVariantStyles = () => {
    switch (variant) {
      case 'card':
        return 'rounded-lg';
      case 'circle':
        return 'rounded-full';
      case 'text':
        return 'rounded';
      default:
        return 'rounded';
    }
  };

  const skeletons = Array.from({ length: count }, (_, index) => (
    <motion.div
      key={index}
      className={`${width} ${height} ${getVariantStyles()} bg-gray-200 dark:bg-miami-neutral-700 relative overflow-hidden ${className} ${index > 0 ? 'mt-2' : ''}`}
      variants={shouldReduceMotion ? pulseVariants : undefined}
      initial="initial"
      animate="animate"
    >
      {!shouldReduceMotion && (
        <motion.div
          className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 dark:via-white/10 to-transparent"
          variants={shimmerVariants}
          initial="initial"
          animate="animate"
        />
      )}
    </motion.div>
  ));

  return count === 1 ? skeletons[0] : <div className="space-y-2">{skeletons}</div>;
};

// Pre-built skeleton components for common use cases
export const ProjectCardSkeleton = () => (
  <div className="p-6 bg-white dark:bg-miami-neutral-800 rounded-lg border border-gray-200 dark:border-miami-neutral-700">
    <div className="flex items-center gap-2 mb-3">
      <LoadingSkeleton width="w-16" height="h-4" variant="text" />
      <LoadingSkeleton width="w-12" height="h-4" variant="text" />
    </div>
    <LoadingSkeleton width="w-3/4" height="h-6" variant="text" className="mb-2" />
    <LoadingSkeleton width="w-1/2" height="h-4" variant="text" className="mb-4" />
    <LoadingSkeleton width="w-full" height="h-16" variant="text" className="mb-4" />
    <div className="flex gap-2 mb-4">
      <LoadingSkeleton width="w-16" height="h-6" variant="text" />
      <LoadingSkeleton width="w-20" height="h-6" variant="text" />
      <LoadingSkeleton width="w-14" height="h-6" variant="text" />
    </div>
    <div className="flex gap-3">
      <LoadingSkeleton width="w-24" height="h-8" variant="text" />
      <LoadingSkeleton width="w-20" height="h-8" variant="text" />
    </div>
  </div>
);

export const TextSkeleton = ({ lines = 3 }) => (
  <div className="space-y-2">
    {Array.from({ length: lines }, (_, index) => (
      <LoadingSkeleton 
        key={index}
        width={index === lines - 1 ? 'w-3/4' : 'w-full'} 
        height="h-4" 
        variant="text" 
      />
    ))}
  </div>
);

export const AvatarSkeleton = ({ size = 'w-12 h-12' }) => (
  <LoadingSkeleton width={size} height={size} variant="circle" />
);

export const ButtonSkeleton = () => (
  <LoadingSkeleton width="w-24" height="h-10" variant="text" />
);

export default LoadingSkeleton;