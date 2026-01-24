import type { APIRoute, GetStaticPaths } from "astro";
import { getCollection } from "astro:content";
import satori from "satori";
import sharp from "sharp";

export const getStaticPaths: GetStaticPaths = async () => {
  const posts = await getCollection("blog");
  const projects = await getCollection("projects");

  const blogPaths = posts.map((post) => ({
    params: { slug: `blog/${post.slug}` },
    props: { title: post.data.title, type: "Blog" },
  }));

  const projectPaths = projects.map((project) => ({
    params: { slug: `projects/${project.slug}` },
    props: { title: project.data.title, type: "Project" },
  }));

  return [
    { params: { slug: "default" }, props: { title: "Pratham Patel", type: "Portfolio" } },
    ...blogPaths,
    ...projectPaths,
  ];
};

export const GET: APIRoute = async ({ props }) => {
  const { title, type } = props as { title: string; type: string };

  // Fetch Inter font
  const fontData = await fetch(
    "https://cdn.jsdelivr.net/fontsource/fonts/inter@latest/latin-600-normal.woff"
  ).then((res) => res.arrayBuffer());

  const svg = await satori(
    {
      type: "div",
      props: {
        style: {
          height: "100%",
          width: "100%",
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          padding: "60px",
          background: "linear-gradient(135deg, #0f172a 0%, #1e293b 100%)",
          fontFamily: "Inter",
        },
        children: [
          {
            type: "div",
            props: {
              style: {
                display: "flex",
                alignItems: "center",
                marginBottom: "40px",
              },
              children: [
                {
                  type: "div",
                  props: {
                    style: {
                      background: "#0ea5e9",
                      color: "white",
                      padding: "8px 16px",
                      borderRadius: "6px",
                      fontSize: "20px",
                      fontWeight: "600",
                    },
                    children: type,
                  },
                },
              ],
            },
          },
          {
            type: "div",
            props: {
              style: {
                fontSize: title.length > 40 ? "48px" : "56px",
                fontWeight: "600",
                color: "white",
                lineHeight: 1.2,
                marginBottom: "40px",
              },
              children: title,
            },
          },
          {
            type: "div",
            props: {
              style: {
                display: "flex",
                alignItems: "center",
                marginTop: "auto",
              },
              children: [
                {
                  type: "div",
                  props: {
                    style: {
                      fontSize: "24px",
                      color: "#94a3b8",
                    },
                    children: "prathamp.com",
                  },
                },
              ],
            },
          },
        ],
      },
    },
    {
      width: 1200,
      height: 630,
      fonts: [
        {
          name: "Inter",
          data: fontData,
          weight: 600,
          style: "normal",
        },
      ],
    }
  );

  const png = await sharp(Buffer.from(svg)).png().toBuffer();

  return new Response(new Uint8Array(png), {
    headers: {
      "Content-Type": "image/png",
      "Cache-Control": "public, max-age=31536000, immutable",
    },
  });
};
