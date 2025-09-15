import React, { useState } from 'react';
import { Rss, Download, Copy, Check, ExternalLink } from 'lucide-react';
import { motion } from 'framer-motion';
import Button from './Button';
import Card from './Card';
import { generateRSSFeed, downloadRSSFeed, copyRSSFeed } from '../../utils/rssGenerator';

const RSSSubscription = ({ className = "" }) => {
  const [copiedFeed, setCopiedFeed] = useState(false);
  const [copiedUrl, setCopiedUrl] = useState(false);

  const rssUrl = 'https://nickogoodis.com/feed.xml';

  const handleCopyFeed = async () => {
    const success = await copyRSSFeed();
    if (success) {
      setCopiedFeed(true);
      setTimeout(() => setCopiedFeed(false), 2000);
    }
  };

  const handleCopyUrl = async () => {
    try {
      await navigator.clipboard.writeText(rssUrl);
      setCopiedUrl(true);
      setTimeout(() => setCopiedUrl(false), 2000);
    } catch (error) {
      console.error('Failed to copy RSS URL:', error);
    }
  };

  const popularRSSReaders = [
    { name: 'Feedly', url: `https://feedly.com/i/subscription/feed/${encodeURIComponent(rssUrl)}` },
    { name: 'Inoreader', url: `https://www.inoreader.com/?add_feed=${encodeURIComponent(rssUrl)}` },
    { name: 'The Old Reader', url: `https://theoldreader.com/feeds/subscribe?url=${encodeURIComponent(rssUrl)}` },
    { name: 'NewsBlur', url: `https://newsblur.com/?url=${encodeURIComponent(rssUrl)}` }
  ];

  return (
    <Card className={`${className}`} padding="lg">
      <div className="flex items-start gap-4">
        <div className="flex-shrink-0">
          <div className="w-12 h-12 bg-orange-500 rounded-lg flex items-center justify-center">
            <Rss className="w-6 h-6 text-white" />
          </div>
        </div>
        
        <div className="flex-1">
          <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-2">
            Subscribe to RSS Feed
          </h3>
          
          <p className="text-gray-600 dark:text-gray-300 mb-4">
            Stay updated with new posts by subscribing to our RSS feed. Get notified whenever we publish new content about software engineering, machine learning, and blockchain technology.
          </p>

          {/* RSS URL Display */}
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              RSS Feed URL:
            </label>
            <div className="flex items-center gap-2">
              <input
                type="text"
                value={rssUrl}
                readOnly
                className="flex-1 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-gray-50 dark:bg-gray-800 text-gray-900 dark:text-white text-sm font-mono"
              />
              <Button
                variant="outline"
                size="sm"
                onClick={handleCopyUrl}
                className="flex-shrink-0"
              >
                {copiedUrl ? (
                  <Check className="w-4 h-4" />
                ) : (
                  <Copy className="w-4 h-4" />
                )}
              </Button>
            </div>
          </div>

          {/* Quick Actions */}
          <div className="flex flex-wrap gap-2 mb-4">
            <Button
              variant="primary"
              size="sm"
              onClick={downloadRSSFeed}
            >
              <Download className="w-4 h-4 mr-2" />
              Download Feed
            </Button>
            
            <Button
              variant="outline"
              size="sm"
              onClick={handleCopyFeed}
            >
              {copiedFeed ? (
                <>
                  <Check className="w-4 h-4 mr-2" />
                  Copied!
                </>
              ) : (
                <>
                  <Copy className="w-4 h-4 mr-2" />
                  Copy Feed XML
                </>
              )}
            </Button>
          </div>

          {/* RSS Reader Links */}
          <div>
            <span className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2 block">
              Or subscribe directly with your favorite RSS reader:
            </span>
            <div className="flex flex-wrap gap-2">
              {popularRSSReaders.map((reader) => (
                <motion.a
                  key={reader.name}
                  href={reader.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1 px-3 py-1 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 text-sm rounded-md hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  {reader.name}
                  <ExternalLink className="w-3 h-3" />
                </motion.a>
              ))}
            </div>
          </div>
        </div>
      </div>
    </Card>
  );
};

export default RSSSubscription;