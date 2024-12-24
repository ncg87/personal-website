import React from 'react';

const RecentBlogs = () => {
    return (
        <section id="blogs">
            <h2>Recent Blogs</h2>
            <ul>
                {blogs.map((blog) => (
                    <li key={blog.id}>
                        <h3>{blog.title}</h3>
                        <p>{blog.summary}</p>
                    </li>
                ))}
            </ul>
        </section>
    );
};

export default RecentBlogs;
