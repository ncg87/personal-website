import { allBlogPosts } from '../data/blogPosts';

export const generateRSSFeed = () => {
  const siteUrl = 'https://nickogoodis.com';
  const feedUrl = `${siteUrl}/feed.xml`;
  
  // Sort posts by date (newest first)
  const sortedPosts = [...allBlogPosts].sort((a, b) => 
    new Date(b.date) - new Date(a.date)
  );

  const lastBuildDate = new Date().toUTCString();
  const mostRecentPostDate = sortedPosts.length > 0 
    ? new Date(sortedPosts[0].date).toUTCString()
    : lastBuildDate;

  // Helper function to clean content for RSS
  const cleanContent = (content) => {
    return content
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  };

  // Helper function to generate excerpt from content
  const generateExcerpt = (content, excerpt) => {
    if (excerpt) return excerpt;
    
    // Extract first paragraph from content
    const lines = content.split('\n').filter(line => line.trim());
    const firstParagraph = lines.find(line => 
      !line.startsWith('#') && 
      !line.startsWith('```') && 
      line.length > 50
    );
    
    if (firstParagraph) {
      return firstParagraph.substring(0, 200) + '...';
    }
    
    return 'Read the full article for more details.';
  };

  const rssItems = sortedPosts.map(post => {
    const postUrl = `${siteUrl}/posts/${post.slug}`;
    const pubDate = new Date(post.date).toUTCString();
    const cleanTitle = cleanContent(post.title);
    const description = cleanContent(generateExcerpt(post.content, post.excerpt));
    const categories = post.tags.map(tag => `      <category>${cleanContent(tag)}</category>`).join('\n');
    
    return `    <item>
      <title>${cleanTitle}</title>
      <link>${postUrl}</link>
      <guid isPermaLink="true">${postUrl}</guid>
      <description><![CDATA[${description}]]></description>
      <pubDate>${pubDate}</pubDate>
      <author>contact@nickogoodis.com (${cleanContent(post.author)})</author>
${categories}
      <source url="${feedUrl}">Nickolas Goodis - Blog</source>
    </item>`;
  }).join('\n');

  const rssFeed = `<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom" xmlns:content="http://purl.org/rss/1.0/modules/content/">
  <channel>
    <title>Nickolas Goodis - Blog</title>
    <link>${siteUrl}</link>
    <description>Technical blog posts about software engineering, machine learning, blockchain technology, and data science by Nickolas Goodis.</description>
    <language>en-us</language>
    <lastBuildDate>${lastBuildDate}</lastBuildDate>
    <pubDate>${mostRecentPostDate}</pubDate>
    <ttl>1440</ttl>
    <atom:link href="${feedUrl}" rel="self" type="application/rss+xml"/>
    <webMaster>contact@nickogoodis.com (Nickolas Goodis)</webMaster>
    <managingEditor>contact@nickogoodis.com (Nickolas Goodis)</managingEditor>
    <copyright>Copyright ${new Date().getFullYear()} Nickolas Goodis</copyright>
    <generator>Custom RSS Generator v1.0</generator>
    <docs>https://blogs.law.harvard.edu/tech/rss</docs>
    <image>
      <url>${siteUrl}/icons/icon-192x192.png</url>
      <title>Nickolas Goodis - Blog</title>
      <link>${siteUrl}</link>
      <width>192</width>
      <height>192</height>
      <description>Nickolas Goodis Portfolio Logo</description>
    </image>
${rssItems}
  </channel>
</rss>`;

  return rssFeed;
};

// Function to download RSS feed
export const downloadRSSFeed = () => {
  const rssFeed = generateRSSFeed();
  const blob = new Blob([rssFeed], { type: 'application/rss+xml' });
  const url = URL.createObjectURL(blob);
  
  const link = document.createElement('a');
  link.href = url;
  link.download = 'feed.xml';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  
  URL.revokeObjectURL(url);
};

// Function to copy RSS feed to clipboard
export const copyRSSFeed = async () => {
  const rssFeed = generateRSSFeed();
  
  try {
    await navigator.clipboard.writeText(rssFeed);
    return true;
  } catch (error) {
    console.error('Failed to copy RSS feed:', error);
    return false;
  }
};

export default generateRSSFeed;