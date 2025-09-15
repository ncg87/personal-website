// Comprehensive project data for case studies and showcase
export const projects = [
  {
    id: 'blockchain-analytics-api',
    slug: 'blockchain-analytics-api',
    title: 'Blockchain Analytics API',
    subtitle: 'High-Performance Multi-Chain Transaction Analysis Platform',
    category: 'Backend Engineering',
    status: 'In Development',
    featured: true,
    timeline: '6 months',
    role: 'Full-stack Developer',
    team: 'Solo',
    
    overview: {
      challenge: 'Build a scalable API capable of processing and analyzing millions of blockchain transactions daily across multiple networks with real-time and historical data support.',
      solution: 'Architected a high-performance system using Rust for data ingestion, Python for analysis, and multiple database technologies optimized for different query patterns.',
      impact: 'Processes millions of transactions daily with sub-second query response times and 99.9% uptime.'
    },

    technologies: ['Python', 'Rust', 'PostgreSQL', 'MongoDB', 'Neo4j', 'AWS'],
    
    keyFeatures: [
      'Real-time transaction ingestion from 7+ blockchain networks',
      'Historical data analysis with optimized query endpoints',
      'Exchange identification and pattern recognition',
      'Scalable microservices architecture',
      'RESTful API with comprehensive documentation'
    ],

    metrics: {
      'Daily Transactions': 'Millions',
      'Response Time': '< 200ms',
      'Uptime': '99.9%',
      'Networks Supported': '7+',
      'Data Storage': 'Multi-TB'
    },

    technicalChallenges: [
      {
        challenge: 'High-volume data ingestion',
        solution: 'Implemented Rust-based ingestion service with async processing and connection pooling',
        result: 'Achieved 10x performance improvement over initial Python implementation'
      },
      {
        challenge: 'Complex relationship queries',
        solution: 'Used Neo4j graph database for transaction relationship analysis',
        result: 'Enabled complex pattern detection with sub-second query times'
      },
      {
        challenge: 'Multi-network data normalization',
        solution: 'Created unified data schema with network-specific adapters',
        result: 'Seamless cross-chain analysis capabilities'
      }
    ],

    links: {
      github: 'https://github.com/ncg87/cryptoflows',
      website: 'https://cryptoflows.ai',
      demo: null
    }
  },

  {
    id: 'poker-algorithm-cfr',
    slug: 'poker-algorithm-cfr',
    title: 'Game Theory Optimal Poker Algorithm',
    subtitle: 'AI-Powered Poker Strategy Using Counterfactual Regret Minimization',
    category: 'Machine Learning',
    status: 'Completed',
    featured: true,
    timeline: '3 months',
    role: 'Algorithm Developer',
    team: 'Solo',

    overview: {
      challenge: 'Develop a poker AI that can achieve near-optimal play through game theory principles, specifically implementing Counterfactual Regret Minimization to reach Nash equilibrium.',
      solution: 'Built a Rust-based CFR implementation with iterative strategy refinement, extensive training protocols, and balanced strategy optimization.',
      impact: 'Created a mathematically sound poker algorithm that minimizes exploitability and achieves optimal play in two-player scenarios.'
    },

    technologies: ['Rust', 'Game Theory', 'CFR Algorithm', 'Nash Equilibrium'],

    keyFeatures: [
      'Counterfactual Regret Minimization implementation',
      'Iterative strategy refinement system',
      'Nash equilibrium convergence',
      'Multi-threaded training for performance',
      'Strategy evaluation and testing framework'
    ],

    metrics: {
      'Algorithm': 'CFR',
      'Convergence': 'Nash Equilibrium',
      'Performance': 'Near-optimal',
      'Training Speed': 'Multi-threaded',
      'Exploitability': 'Minimized'
    },

    technicalChallenges: [
      {
        challenge: 'Memory efficiency for large game trees',
        solution: 'Implemented abstraction techniques and efficient data structures',
        result: 'Reduced memory usage by 70% while maintaining accuracy'
      },
      {
        challenge: 'Convergence speed optimization',
        solution: 'Applied Monte Carlo sampling and parallel processing',
        result: 'Achieved 5x faster convergence to Nash equilibrium'
      },
      {
        challenge: 'Strategy balance verification',
        solution: 'Built comprehensive testing framework with mathematical validation',
        result: 'Verified optimal play against various opponent strategies'
      }
    ],

    links: {
      github: 'https://github.com/ncg87/poker-cfr',
      website: null,
      demo: null
    }
  },

  {
    id: 'pairs-trading-algorithm',
    slug: 'pairs-trading-algorithm',
    title: 'Pairs Trading Statistical Arbitrage',
    subtitle: 'ML-Powered Quantitative Trading with 600% Returns',
    category: 'Quantitative Finance',
    status: 'Backtested',
    featured: true,
    timeline: '4 months',
    role: 'Quantitative Developer',
    team: 'Solo',

    overview: {
      challenge: 'Develop a systematic trading algorithm that identifies and capitalizes on price divergences between correlated financial instruments using machine learning.',
      solution: 'Built a comprehensive system using PyTorch for ML models, extensive backtesting framework, and live paper trading integration with robust risk management.',
      impact: 'Achieved 600% returns with 2.8 Sharpe ratio over 10-year backtest period, demonstrating consistent alpha generation.'
    },

    technologies: ['Python', 'PyTorch', 'Pandas', 'AWS', 'Alpaca API', 'Yahoo Finance'],

    keyFeatures: [
      'Machine learning pair selection algorithm',
      'Statistical arbitrage signal generation',
      'Comprehensive backtesting framework',
      'Real-time market data integration',
      'Risk management and position sizing',
      'Live paper trading implementation'
    ],

    metrics: {
      'Returns': '600% (10-year)',
      'Sharpe Ratio': '2.8',
      'Max Drawdown': '<15%',
      'Win Rate': '68%',
      'Trading Frequency': 'Daily'
    },

    technicalChallenges: [
      {
        challenge: 'Feature engineering for pair selection',
        solution: 'Developed cointegration analysis and correlation stability metrics',
        result: 'Improved pair selection accuracy by 40%'
      },
      {
        challenge: 'Market regime detection',
        solution: 'Implemented adaptive algorithms for different market conditions',
        result: 'Maintained performance across bull/bear markets'
      },
      {
        challenge: 'Transaction cost modeling',
        solution: 'Built realistic cost models including slippage and fees',
        result: 'Achieved more accurate backtesting results'
      }
    ],

    links: {
      github: 'https://github.com/ncg87/pairs-trading',
      website: null,
      demo: null
    }
  },

  {
    id: 'bitcoin-visualization',
    slug: 'bitcoin-visualization',
    title: 'Bitcoin Transaction Network Visualization',
    subtitle: '3D Interactive Blockchain Explorer',
    category: 'Data Visualization',
    status: 'Live',
    featured: false,
    timeline: '2 months',
    role: 'Frontend Developer',
    team: 'Solo',

    overview: {
      challenge: 'Create an intuitive way to explore and understand Bitcoin transaction relationships through interactive 3D visualization.',
      solution: 'Built a web-based 3D visualization using Three.js and React, integrated with CryptoFlows.ai backend for real-time blockchain data.',
      impact: 'Provides researchers and enthusiasts with an accessible tool to explore Bitcoin transaction patterns and network topology.'
    },

    technologies: ['React', 'Three.js', 'D3.js', 'WebGL', 'JavaScript'],

    keyFeatures: [
      '3D interactive transaction graph',
      'Real-time blockchain data integration',
      'Address clustering and analysis',
      'Transaction flow visualization',
      'Responsive web interface',
      'Export capabilities for research'
    ],

    metrics: {
      'Visualization': '3D Interactive',
      'Data Source': 'Real-time API',
      'Performance': '60fps',
      'Compatibility': 'Modern browsers',
      'User Engagement': 'High'
    },

    links: {
      github: 'https://github.com/ncg87/bitcoin-tx-graph',
      website: 'https://bitcoin-tx-graph.vercel.app/',
      demo: 'https://bitcoin-tx-graph.vercel.app/'
    }
  },

  {
    id: 'ai-researcher',
    slug: 'ai-researcher',
    title: 'Automated AI Researcher',
    subtitle: 'LLM-Powered Research Assistant for Academic Papers',
    category: 'AI Tools',
    status: 'Open Source',
    featured: false,
    timeline: '1 month',
    role: 'AI Engineer',
    team: 'Solo',

    overview: {
      challenge: 'Automate the time-consuming process of literature review and research analysis for academic papers and research queries.',
      solution: 'Created an intelligent CLI tool that leverages multiple LLM APIs to analyze arXiv papers, expand research queries, and provide structured insights.',
      impact: 'Reduces research time by 80% while providing comprehensive analysis and structured output for academic work.'
    },

    technologies: ['Python', 'OpenAI API', 'Rich CLI', 'arXiv API', 'LangChain'],

    keyFeatures: [
      'Multi-LLM integration for diverse perspectives',
      'Automated arXiv paper analysis',
      'Research query expansion and refinement',
      'Real-time progress tracking with Rich CLI',
      'Structured research output management',
      'Citation and reference handling'
    ],

    metrics: {
      'Time Savings': '80%',
      'LLM Integration': 'Multiple APIs',
      'Paper Analysis': 'Automated',
      'Output Format': 'Structured',
      'User Interface': 'Interactive CLI'
    },

    links: {
      github: 'https://github.com/ncg87/Research-Assistant',
      website: null,
      demo: null
    }
  },

  {
    id: 'donation-blockchain',
    slug: 'donation-blockchain',
    title: 'Donation Blockchain Portal',
    subtitle: 'Decentralized Donation Platform with Smart Contracts',
    category: 'Blockchain Development',
    status: 'Deployed',
    featured: false,
    timeline: '6 weeks',
    role: 'Blockchain Developer',
    team: 'Solo',

    overview: {
      challenge: 'Create a transparent, secure donation platform using blockchain technology with donor tier tracking and secure fund management.',
      solution: 'Built a full-stack DApp using Solidity smart contracts, React frontend, and Web3 integration for secure ETH transactions.',
      impact: 'Enables transparent charitable donations with immutable records and automated tier-based recognition system.'
    },

    technologies: ['Solidity', 'React', 'Web3.js', 'Foundry', 'Node.js', 'Vercel'],

    keyFeatures: [
      'Smart contract-based donation handling',
      'Automated donor tier recognition',
      'Secure ETH transaction processing',
      'Owner-only fund withdrawal system',
      'Transparent donation tracking',
      'Mobile-responsive interface'
    ],

    metrics: {
      'Security': 'Smart Contract',
      'Blockchain': 'Ethereum',
      'Deployment': 'Live on Vercel',
      'Transaction Type': 'ETH',
      'Access Control': 'Owner-based'
    },

    links: {
      github: 'https://github.com/ncg87/donation-app',
      website: 'https://donation-app-roan.vercel.app/',
      demo: 'https://donation-app-roan.vercel.app/'
    }
  },

  {
    id: 'dex-tracker',
    slug: 'dex-tracker',
    title: 'DEX Tracker System',
    subtitle: 'Granular Decentralized Exchange Transaction Monitoring',
    category: 'Blockchain Analytics',
    status: 'Completed',
    featured: false,
    timeline: '3 months',
    role: 'Backend Developer',
    team: 'Solo',

    overview: {
      challenge: 'Track and store every transaction on multiple decentralized exchanges (DEXs) at the most granular level for comprehensive DeFi analysis.',
      solution: 'Built a comprehensive tracking system that monitors DEX protocols, captures transaction data, and stores structured information in a scalable database.',
      impact: 'Provides detailed insights into DeFi trading patterns, liquidity flows, and market behavior across multiple exchanges.'
    },

    technologies: ['Python', 'Web3.py', 'PostgreSQL', 'Docker', 'AWS'],

    keyFeatures: [
      'Multi-DEX protocol integration',
      'Real-time transaction monitoring',
      'Granular data capture and storage',
      'Historical data analysis capabilities',
      'Scalable database architecture',
      'RESTful API for data access'
    ],

    metrics: {
      'DEXs Tracked': 'Multiple',
      'Data Granularity': 'Transaction-level',
      'Storage': 'PostgreSQL',
      'Real-time': 'Yes',
      'Coverage': 'Comprehensive'
    },

    links: {
      github: 'https://github.com/ncg87/dex_tracker_system',
      website: null,
      demo: null
    }
  },

  {
    id: 'personal-website',
    slug: 'personal-website',
    title: 'Personal Portfolio Website',
    subtitle: 'Modern React Portfolio with Design System',
    category: 'Frontend Development',
    status: 'Live',
    featured: false,
    timeline: '4 weeks',
    role: 'Full-stack Developer',
    team: 'Solo',

    overview: {
      challenge: 'Create a professional portfolio website that showcases projects, experience, and skills with modern web technologies and best practices.',
      solution: 'Built a responsive React application with Tailwind CSS, featuring a custom design system, SEO optimization, and comprehensive project showcases.',
      impact: 'Serves as a professional online presence demonstrating technical skills and project portfolio to potential employers and collaborators.'
    },

    technologies: ['React', 'Tailwind CSS', 'Framer Motion', 'Vercel', 'Vite'],

    keyFeatures: [
      'Responsive design system',
      'SEO optimization with meta tags',
      'Project case study pages',
      'Resume integration with PDF download',
      'Smooth animations and transitions',
      'Accessibility compliance'
    ],

    metrics: {
      'Framework': 'React 18',
      'Styling': 'Tailwind CSS',
      'Performance': 'Optimized',
      'SEO': 'Complete',
      'Accessibility': 'WCAG 2.1 AA'
    },

    links: {
      github: 'https://github.com/ncg87/personal-website',
      website: 'https://nickolasgoodis.com',
      demo: 'https://nickolasgoodis.com'
    }
  },

  {
    id: 'ml-deployment-platform',
    slug: 'ml-deployment-platform',
    title: 'Machine Learning Deployment Platform',
    subtitle: 'Multi-Model ML Showcase with Docker & AWS',
    category: 'Machine Learning',
    status: 'Deployed',
    featured: false,
    timeline: '2 months',
    role: 'ML Engineer',
    team: 'Solo',

    overview: {
      challenge: 'Create a platform to showcase multiple machine learning models including image captioning and multilingual translation with production-ready deployment.',
      solution: 'Architected a Flask-based web application with Docker containerization, featuring multiple trained ML models with interactive interfaces.',
      impact: 'Demonstrates practical ML model deployment skills and provides accessible interfaces for testing computer vision and NLP capabilities.'
    },

    technologies: ['Python', 'Flask', 'PyTorch', 'Docker', 'AWS', 'Nginx'],

    keyFeatures: [
      'Multiple ML model integration',
      'Image captioning with computer vision',
      'Multilingual translation capabilities',
      'Interactive web interfaces',
      'Docker containerization',
      'AWS cloud deployment'
    ],

    metrics: {
      'Models Deployed': '6+',
      'Languages Supported': '6',
      'Framework': 'Flask + PyTorch',
      'Deployment': 'AWS + Docker',
      'Interface': 'Web-based'
    },

    links: {
      github: 'https://github.com/ncg87/projects-website',
      website: 'https://nickogoodis.com',
      demo: null
    }
  },

  {
    id: 'face-detector-yolo',
    slug: 'face-detector-yolo',
    title: 'Real-Time Face Detection with YOLO',
    subtitle: 'Custom YOLO Model for Live Video Face Detection',
    category: 'Computer Vision',
    status: 'Completed',
    featured: false,
    timeline: '3 weeks',
    role: 'ML Engineer',
    team: 'Solo',

    overview: {
      challenge: 'Develop a real-time face detection system capable of identifying faces in live video streams with high accuracy and performance.',
      solution: 'Implemented and trained a custom YOLO (You Only Look Once) model optimized for face detection with real-time processing capabilities.',
      impact: 'Achieved real-time face detection suitable for applications in security, user interaction, and computer vision research.'
    },

    technologies: ['Python', 'YOLO', 'OpenCV', 'PyTorch', 'Computer Vision'],

    keyFeatures: [
      'Real-time face detection in video',
      'Custom YOLO model training',
      'High-performance inference',
      'Live camera integration',
      'Bounding box visualization',
      'Multi-face detection support'
    ],

    metrics: {
      'Model': 'Custom YOLO',
      'Performance': 'Real-time',
      'Accuracy': 'High',
      'Input': 'Live video',
      'Detection': 'Multi-face'
    },

    links: {
      github: 'https://github.com/ncg87/YOLO',
      website: null,
      demo: null
    }
  },

  {
    id: 'multilingual-translation',
    slug: 'multilingual-translation',
    title: 'Multilingual Translation Model',
    subtitle: '6-Language Neural Machine Translation System',
    category: 'Natural Language Processing',
    status: 'Completed',
    featured: false,
    timeline: '6 weeks',
    role: 'ML Engineer',
    team: 'Solo',

    overview: {
      challenge: 'Create a neural machine translation system capable of translating between 6 different languages with high accuracy using transfer learning.',
      solution: 'Developed a transformer-based translation model using transfer learning approaches, training on multiple language pairs with shared representations.',
      impact: 'Enables multilingual communication support for applications requiring real-time translation capabilities across major world languages.'
    },

    technologies: ['Python', 'PyTorch', 'Transformers', 'Hugging Face', 'Transfer Learning'],

    keyFeatures: [
      '6-language translation support',
      'Transfer learning architecture',
      'Bidirectional translation',
      'High translation accuracy',
      'Efficient model inference',
      'Language auto-detection'
    ],

    metrics: {
      'Languages': '6',
      'Architecture': 'Transformer',
      'Training': 'Transfer Learning',
      'Performance': 'High accuracy',
      'Inference': 'Fast'
    },

    links: {
      github: 'https://github.com/ncg87/machine-translation',
      website: null,
      demo: null
    }
  },

  {
    id: 'image-captioning-model',
    slug: 'image-captioning-model',
    title: 'Image Captioning Model',
    subtitle: 'CNN-RNN Architecture for Automatic Image Description',
    category: 'Computer Vision',
    status: 'Completed',
    featured: false,
    timeline: '4 weeks',
    role: 'ML Engineer',
    team: 'Solo',

    overview: {
      challenge: 'Develop an AI system that can automatically generate natural language descriptions for images using deep learning techniques.',
      solution: 'Built a CNN-RNN architecture combining computer vision and natural language processing, trained on the Flickr8k dataset for image captioning.',
      impact: 'Demonstrates understanding of multimodal AI systems combining vision and language, applicable to accessibility tools and content generation.'
    },

    technologies: ['Python', 'PyTorch', 'CNN', 'RNN', 'LSTM', 'Computer Vision'],

    keyFeatures: [
      'Automatic image caption generation',
      'CNN feature extraction',
      'RNN language modeling',
      'Flickr8k dataset training',
      'Natural language output',
      'Attention mechanism integration'
    ],

    metrics: {
      'Architecture': 'CNN-RNN',
      'Dataset': 'Flickr8k',
      'Output': 'Natural language',
      'Training': 'Supervised',
      'Performance': 'High quality'
    },

    links: {
      github: 'https://github.com/ncg87/image-captioning',
      website: null,
      demo: null
    }
  },

  {
    id: 'python-tic-tac-toe',
    slug: 'python-tic-tac-toe',
    title: 'Python Tic-Tac-Toe Game',
    subtitle: 'Interactive Game with AI Opponents of Varying Difficulty',
    category: 'Game Development',
    status: 'Completed',
    featured: false,
    timeline: '1 week',
    role: 'Developer',
    team: 'Solo',

    overview: {
      challenge: 'Create an engaging Tic-Tac-Toe game in Python with multiple playing modes including AI opponents of different difficulty levels.',
      solution: 'Developed a complete game implementation with clean user interface, two-player mode, and AI opponents using minimax algorithm with varying difficulty.',
      impact: 'Demonstrates game development skills, algorithm implementation, and user experience design in a classic game format.'
    },

    technologies: ['Python', 'Game Logic', 'Minimax Algorithm', 'CLI Interface'],

    keyFeatures: [
      'Two-player local gameplay',
      'AI opponents with difficulty levels',
      'Minimax algorithm implementation',
      'Clean command-line interface',
      'Game state management',
      'Win condition detection'
    ],

    metrics: {
      'Game Modes': 'PvP + AI',
      'AI Levels': 'Multiple',
      'Algorithm': 'Minimax',
      'Interface': 'CLI',
      'Language': 'Python'
    },

    links: {
      github: 'https://github.com/ncg87/tic-tac-toe',
      website: null,
      demo: null
    }
  }
];

// Helper functions
export const getFeaturedProjects = () => projects.filter(p => p.featured);
export const getProjectBySlug = (slug) => projects.find(p => p.slug === slug);
export const getProjectsByCategory = (category) => projects.filter(p => p.category === category);
export const getAllCategories = () => [...new Set(projects.map(p => p.category))];