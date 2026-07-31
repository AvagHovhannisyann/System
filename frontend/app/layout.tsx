import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

import { Providers } from "@/components/providers";
import { SidebarNav } from "@/components/sidebar-nav";

const geistSans = Geist({
  variable: "--font-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Quant Research Platform",
  description:
    "Operator dashboard for the systematic equity research platform (paper trading only)",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        <Providers>
          <div className="flex min-h-screen">
            <aside className="w-56 shrink-0 border-r">
              <div className="border-b p-4 text-sm font-semibold">
                Quant Research Platform
              </div>
              <SidebarNav />
            </aside>
            <main className="flex-1 p-6">{children}</main>
          </div>
        </Providers>
      </body>
    </html>
  );
}
