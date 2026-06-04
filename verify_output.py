import os
output_dir = r"D:\Polythecninco di Milano\AFB_Lab\AS_Lab_Combined"
md_files = [f for f in os.listdir(output_dir) if f.endswith('.md')]
pdf_files = os.listdir(os.path.join(output_dir, 'PDFs'))

print(f"📦 Output Summary:")
print(f"{'='*50}")
print(f"✅ Markdown files: {len(md_files)}")
print(f"✅ PDF files: {len(pdf_files)}")
print(f"\n📋 Sample markdown files:")
for f in sorted(md_files)[:5]:
    print(f"   • {f}")
if len(md_files) > 5:
    print(f"   ... and {len(md_files) - 5} more")
print(f"\n📍 Location: {output_dir}")
