import type { CollectionEntry } from "astro:content";

export interface AdjacentLink {
  href: string;
  title: string;
}

export interface AdjacentNavigation {
  previous?: AdjacentLink;
  next?: AdjacentLink;
}

type TitledEntry = {
  slug: string;
  data: {
    title: string;
  };
};

export function sortBlogEntries(entries: CollectionEntry<"blog">[]) {
  return [...entries]
    .filter(({ data }) => !data.draft)
    .sort(
      (a, b) =>
        b.data.date.valueOf() - a.data.date.valueOf() ||
        (a.data.order ?? 0) - (b.data.order ?? 0),
    );
}

export function sortProjectEntries(entries: CollectionEntry<"projects">[]) {
  return [...entries]
    .filter(({ data }) => !data.draft)
    .sort((a, b) => a.data.title.localeCompare(b.data.title));
}

export function sortNoteEntries(entries: CollectionEntry<"notes">[]) {
  return [...entries]
    .filter(({ data }) => !data.draft)
    .sort((a, b) => b.data.date.valueOf() - a.data.date.valueOf());
}

export function getAdjacentNavigation<T extends TitledEntry>(
  entries: T[],
  currentSlug: string,
  basePath: string,
): AdjacentNavigation {
  const currentIndex = entries.findIndex((entry) => entry.slug === currentSlug);

  if (currentIndex === -1) {
    return {};
  }

  const normalizedBasePath = basePath.endsWith("/")
    ? basePath.slice(0, -1)
    : basePath;
  const previousEntry = currentIndex > 0 ? entries[currentIndex - 1] : undefined;
  const nextEntry =
    currentIndex < entries.length - 1 ? entries[currentIndex + 1] : undefined;

  return {
    previous: previousEntry && {
      href: `${normalizedBasePath}/${previousEntry.slug}`,
      title: previousEntry.data.title,
    },
    next: nextEntry && {
      href: `${normalizedBasePath}/${nextEntry.slug}`,
      title: nextEntry.data.title,
    },
  };
}
