import React from 'react';
import { Download, ExternalLink, Mail, Phone, MapPin } from 'lucide-react';
import Button from './ui/Button';
import Card from './ui/Card';
import Badge from './ui/Badge';
import SEO from './SEO';

const Resume = () => {
  const handleDownloadPDF = () => {
    // Track download event for analytics
    if (window.gtag) {
      window.gtag('event', 'resume_download', {
        event_category: 'engagement',
        event_label: 'PDF Download'
      });
    }
    
    // Trigger download
    const link = document.createElement('a');
    link.href = '/resume.pdf';
    link.download = 'Nickolas_Goodis_Resume.pdf';
    link.click();
  };

  const contact = {
    name: "Nickolas Charles Goodis",
    permanentAddress: "161 Palmer Ave, Winter Park, FL 32789",
    schoolAddress: "1320 South Dixie Hwy, Coral Gables, FL 33146",
    phone: "(505) 297-8357",
    email: "ncg87@miami.edu",
    linkedin: "LinkedIn",
    github: "Github",
    portfolio: "Portfolio"
  };

  const education = {
    school: "University of Miami",
    location: "Coral Gables, Florida",
    period: "August 2022 - May 2026",
    degrees: [
      "Bachelor of Science in Computer Science",
      "Bachelor of Science in Mathematics", 
      "Bachelor of Science in Data Science and Artificial Intelligence"
    ],
    gpa: "3.9",
    sat: "1520/1600",
    coursework: [
      "Probability Theory", "Modern Algebra", "Machine Learning", "Software Engineering",
      "Theory of Computing", "Data Structure and Algorithm Analysis", "Statistics Theory",
      "Real Analysis", "Operating Systems", "Computer Networks", "Linear Algebra"
    ]
  };

  const experience = [
    {
      title: "Nuclear Data Science Intern",
      company: "NextEra Energy",
      location: "Juno Beach, Florida",
      period: "August 2025 – Present",
      responsibilities: [
        "Built and deployed MCP server enabling the retrieval of operational data from Nuclear Postgres database for enterprise AI assistant",
        "Assisted in developing, optimizing, and developing a ReAct AI agent for automating nuclear MSPI compliance monitoring processes",
        "Enhanced security and performance across multiple AI model repositories for production deployment"
      ]
    },
    {
      title: "Undergraduate Researcher",
      company: "University of Miami",
      location: "Coral Gables, Florida", 
      period: "January 2024 – Present",
      subtitle: "Research under Director of Graduate Studies, Dilip Sarkar, on advanced machine and deep learning applications",
      responsibilities: [
        "Engineered an ensemble of networks boosting N-shot learning performance by up to 15 percent",
        "Implemented a state-of-the art W-Net architecture to generate synthetic masks for medical datasets in PyTorch",
        "Conducted experiments to minimize expert annotations in medical image segmentation, through a semi-automatic approach",
        "Co-authoring a research paper on the semi-automatic segmentation method, scheduled for publication this year"
      ]
    },
    {
      title: "Quantitative Trader Intern",
      company: "Greenland Risk Management",
      location: "Dallas, Texas",
      period: "August 2024 – December 2024",
      responsibilities: [
        "Developed a news-based futures trading algorithm for cotton, leveraging web scraping, MongoDB, and AWS containerization",
        "Engineered a multi-stage pipeline: web scraping large datasets, classifying relevance, and training LLMs to determine article outcomes",
        "Implemented predictive trading based on LLM outputs, demonstrating proficiency in end-to-end cloud ML pipeline development"
      ]
    }
  ];

  const projects = [
    {
      name: "Blockchain Analytics API",
      technologies: ["Python", "Rust", "PostgreSQL", "MongoDB", "Neo4j", "AWS"],
      description: [
        "Architected and developed a high-performance API to store and analyze millions of blockchain transactions per day across all EVM compatible networks (Base, Ethereum, BNB), Bitcoin, Solana, and XRP, with support for both real-time and historical data update",
        "Designed scalable and optimized endpoints to enable exchange identification and transaction pattern analysis for external application"
      ]
    },
    {
      name: "Game Theory Optimal Poker Algorithm",
      technologies: ["Rust"],
      description: [
        "Implemented Counterfactual Regret Minimization (CFR) to iteratively refine strategy, resulting in a near-optimal poker algorithm",
        "Achieved Nash equilibrium through extensive training, creating a balanced strategy that minimizes exploitability"
      ]
    },
    {
      name: "Automated AI Researcher",
      technologies: ["Python"],
      description: [
        "Leveraged multiple LLM API's for determining and analyzing arXiv research papers, to develop and expand upon a research query",
        "Designed an interactive CLI with Rich library providing real-time progress tracking, and structured research result management"
      ]
    },
    {
      name: "Pairs Trading Statistical Arbitrage Optimizer",
      technologies: ["Python", "AWS"],
      description: [
        "Engineered a predictive model using PyTorch, Pandas, and Yahoo Finance to identify optimal entry points for pairs trading arbitrage",
        "Validated model efficacy through extensive back testing, achieving a 600% return and 2.8 Sharpe Ratio over a 10-year period",
        "Implemented live paper trading functionality via Alpaca API and AWS, enabling daily market participation"
      ]
    },
    {
      name: "Personal Website",
      technologies: ["React", "JavaScript", "HTML/CSS", "Vercel"],
      hasLink: true,
      description: [
        "Designed and developed a personal website showcasing projects and experience, ensuring compatibility on all types of devices"
      ]
    }
  ];

  const skills = {
    languages: ["Python", "Java", "C", "C++", "JavaScript", "Rust", "HTML/CSS", "SQL", "Solidity", "TypeScript", "PERL"],
    tools: ["Github", "Cursor", "VSCode", "AWS", "Linux", "Docker", "Neo4j", "Nginx", "Android Studio", "MongoDB", "Vercel", "Foundry", "PostgreSQL"],
    frameworks: ["PyTorch", "Sci-Kit Learn", "NumPy", "Pandas", "Flask", "React", "Node.js", "Next.js"]
  };

  return (
    <>
      <SEO 
        title="Resume - Nickolas Goodis"
        description="Software engineer resume featuring React, TypeScript, Python, and machine learning experience. Download PDF version available."
        keywords="resume, CV, software engineer, data scientist, React, TypeScript, Python, machine learning, University of Miami"
        url="https://nickogoodis.com/resume"
      />
      <div className="min-h-screen bg-gray-50 dark:bg-miami-neutral-900 py-8">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header with Download Button */}
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-8 no-print">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">Resume</h1>
            <p className="text-gray-600 dark:text-gray-300">Software Engineer & Data Scientist</p>
          </div>
          <Button onClick={handleDownloadPDF} className="mt-4 sm:mt-0">
            <Download className="w-4 h-4 mr-2" />
            Download PDF
          </Button>
        </div>

        {/* Resume Content */}
        <Card padding="lg" className="resume-content" id="main-content">
          {/* Header */}
          <header className="text-center border-b border-gray-200 dark:border-gray-700 pb-6 mb-8">
            <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
              {contact.name}
            </h1>
            <div className="flex flex-col sm:flex-row justify-center items-center gap-4 text-sm text-gray-600 dark:text-gray-300">
              <div className="flex items-center gap-1">
                <MapPin className="w-4 h-4" />
                <span>{contact.permanentAddress}</span>
              </div>
              <div className="flex items-center gap-1">
                <Phone className="w-4 h-4" />
                <span>{contact.phone}</span>
              </div>
              <div className="flex items-center gap-1">
                <Mail className="w-4 h-4" />
                <a href={`mailto:${contact.email}`} className="text-miami-green-600 hover:text-miami-green-700">
                  {contact.email}
                </a>
              </div>
            </div>
            <div className="flex justify-center gap-4 mt-2 text-sm">
              <a href="#" className="text-miami-green-600 hover:text-miami-green-700 flex items-center gap-1">
                <ExternalLink className="w-3 h-3" />
                LinkedIn
              </a>
              <a href="#" className="text-miami-green-600 hover:text-miami-green-700 flex items-center gap-1">
                <ExternalLink className="w-3 h-3" />
                Github
              </a>
              <a href="#" className="text-miami-green-600 hover:text-miami-green-700 flex items-center gap-1">
                <ExternalLink className="w-3 h-3" />
                Portfolio
              </a>
            </div>
          </header>

          {/* Education */}
          <section className="mb-8 print-no-break">
            <h2 className="text-xl font-semibold text-miami-green-600 dark:text-miami-green-400 mb-4 border-b border-miami-green-200 dark:border-miami-green-800 pb-2">
              Education
            </h2>
            <div className="mb-4">
              <div className="flex flex-col sm:flex-row sm:justify-between sm:items-start mb-2">
                <h3 className="font-semibold text-lg text-gray-900 dark:text-white">{education.school}</h3>
                <span className="text-sm text-gray-600 dark:text-gray-300">{education.location}, {education.period}</span>
              </div>
              {education.degrees.map((degree, index) => (
                <p key={index} className="text-gray-700 dark:text-gray-300 italic">{degree}</p>
              ))}
              <div className="mt-2 flex flex-wrap gap-4 text-sm">
                <span className="font-medium">GPA: {education.gpa}</span>
                <span className="font-medium">SAT: {education.sat}</span>
              </div>
              <div className="mt-3">
                <p className="font-medium text-gray-900 dark:text-white mb-2">Relevant Coursework:</p>
                <div className="flex flex-wrap gap-1">
                  {education.coursework.map((course, index) => (
                    <Badge key={index} variant="default" size="xs">
                      {course}
                    </Badge>
                  ))}
                </div>
              </div>
            </div>
          </section>

          {/* Professional Experience */}
          <section className="mb-8">
            <h2 className="text-xl font-semibold text-miami-green-600 dark:text-miami-green-400 mb-4 border-b border-miami-green-200 dark:border-miami-green-800 pb-2">
              Professional Experience
            </h2>
            {experience.map((job, index) => (
              <div key={index} className="mb-6 print-no-break">
                <div className="flex flex-col sm:flex-row sm:justify-between sm:items-start mb-2">
                  <div>
                    <h3 className="font-semibold text-lg text-gray-900 dark:text-white">{job.title}</h3>
                    <p className="text-miami-green-600 dark:text-miami-green-400 font-medium">{job.company}</p>
                    {job.subtitle && <p className="text-sm text-gray-600 dark:text-gray-300 italic mt-1">{job.subtitle}</p>}
                  </div>
                  <span className="text-sm text-gray-600 dark:text-gray-300">{job.location}, {job.period}</span>
                </div>
                <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
                  {job.responsibilities.map((resp, idx) => (
                    <li key={idx}>{resp}</li>
                  ))}
                </ul>
              </div>
            ))}
          </section>

          {/* Projects */}
          <section className="mb-8">
            <h2 className="text-xl font-semibold text-miami-green-600 dark:text-miami-green-400 mb-4 border-b border-miami-green-200 dark:border-miami-green-800 pb-2">
              Projects
            </h2>
            {projects.map((project, index) => (
              <div key={index} className="mb-6 print-no-break">
                <div className="flex flex-wrap items-center gap-2 mb-2">
                  <h3 className="font-semibold text-lg text-gray-900 dark:text-white">{project.name}</h3>
                  <span className="text-gray-400">|</span>
                  <div className="flex flex-wrap gap-1">
                    {project.technologies.map((tech, idx) => (
                      <Badge key={idx} variant="primary" size="xs">
                        {tech}
                      </Badge>
                    ))}
                  </div>
                  {project.hasLink && (
                    <a href="#" className="text-miami-green-600 hover:text-miami-green-700 text-sm flex items-center gap-1">
                      <ExternalLink className="w-3 h-3" />
                      Link
                    </a>
                  )}
                </div>
                <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
                  {project.description.map((desc, idx) => (
                    <li key={idx}>{desc}</li>
                  ))}
                </ul>
              </div>
            ))}
          </section>

          {/* Technical Skills */}
          <section className="mb-8">
            <h2 className="text-xl font-semibold text-miami-green-600 dark:text-miami-green-400 mb-4 border-b border-miami-green-200 dark:border-miami-green-800 pb-2">
              Technical Skills
            </h2>
            <div className="space-y-4">
              <div>
                <h3 className="font-medium text-gray-900 dark:text-white mb-2">Languages:</h3>
                <div className="flex flex-wrap gap-1">
                  {skills.languages.map((lang, index) => (
                    <Badge key={index} variant="default" size="sm">
                      {lang}
                    </Badge>
                  ))}
                </div>
              </div>
              <div>
                <h3 className="font-medium text-gray-900 dark:text-white mb-2">Developer Tools:</h3>
                <div className="flex flex-wrap gap-1">
                  {skills.tools.map((tool, index) => (
                    <Badge key={index} variant="default" size="sm">
                      {tool}
                    </Badge>
                  ))}
                </div>
              </div>
              <div>
                <h3 className="font-medium text-gray-900 dark:text-white mb-2">Libraries/Frameworks:</h3>
                <div className="flex flex-wrap gap-1">
                  {skills.frameworks.map((framework, index) => (
                    <Badge key={index} variant="default" size="sm">
                      {framework}
                    </Badge>
                  ))}
                </div>
              </div>
            </div>
          </section>
        </Card>
      </div>
      </div>
    </>
  );
};

export default Resume;