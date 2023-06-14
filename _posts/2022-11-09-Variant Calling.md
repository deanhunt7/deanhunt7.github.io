I've heard a lot about bioinformatics and even talked about it a lot, but I never took the time to actually see what it was about. Yi recently sent us a nice tutorial for some baby bioinformatics stuff, and I followed it, trying to learn about bioinformatics common tools and lingo. This post is just summarizing my learning and any questions I might have.

# Some Background

DNA/RNA sequencing is becoming more and more popular, as capabilities for data processing and gene editing are refined and improved. New sequencing techniques have emerged, allowing quicker, more efficient sequence analysis to be performed on larger groups and samples.

The whole of sequence analysis boils down to sequence alignment and comparison. With a reference sequence (hg38, from the human genome), and a target sequence, you can call variants by aligning the two strings, and comparing differences.

The human genome is long, so long that sequence analysis on its billions of characters is slow and expensive. The *exome* is a smaller subset of the genome, consisting of about 1% of the whole genome. The exome is important, as changes in it have been found to cause more than 80% of genetic diseases.

# The Project

The point of this project was to "call" variants between two parents and a child (proband). The use of "calling" threw me off a bit, but I intuited it by thinking of it in terms of uncertainty. We have to "call" a SNP/indel (single nucleotide polymorphism and insertion/deletion) as a variant, because our reads aren't always accurate. We have to set a statistical threshold, above which we call the change as a variant, and below which we attribute the difference to error.

Each parent and child have 2 read fq files associated with them, R1 and R2. These denote different reads from primers at each end of the genome. Neither of them contain the whole genome, hence the combination of the .sai files in step 5.

## 1. Download sample exome sequences

For each of the the father, mother, and proband, I downloaded two exome sequence files (.fq). These files follow FASTQ format (https://support.illumina.com/bulletins/2016/04/fastq-files-explained.html), kind of the standard format for sequencing outputs. FASTQ consists of a sequence of base-pair readings and quality measurements for each read (denoted by an ASCII character).

## 2. Download human genome reference

The human genome file is hosted by UCSC (who also has a cuttng-edge bioinformatics program) at http://hgdownload.cse.ucsc.edu/goldenpath/hg38/bigZips/. hg38 is simply the most updated reference genome. It downloads as a .fa file, just a huge string of bases.

## 3. Index the human genome

Indexing of the human genome acts just like indexing of a book; it makes queries faster, saving both time and memory. Indexing was performed by BWA (Burrows-Wheeler Aligner), and took about an hour for hg38.

We couldn't just download a pre-indexed genome, because we needed it to be indexed by BWA specifically, since we used the indexed file to align in the next step.

## 4. Align test sequences to reference

Now that we have the indexed reference genome, we can align all of our test genomes to it. This outputs a sai file, which are later aligned to become a SAM file.

I'm actually not really sure the use of multiple sai files. How do we combine the two sai files?

## 5. Combine multiple sai aligned sequences for the same genome

You aligned multiple reads of the same test genomes (a few for each), and now you want to combine them to a single sequence alignment file. A SAM file shows you your sequences aligned to a reference sequence. This helps for variant calling, by easily being able to compare different bases to find variants.

Usually, SAM files are used for humans, and BAM files are used for computation (binary SAM file). 

## 6. Use picard to sort the samples, mark duplicates, and convert to .mpileup format

Picard is used to sort the samples by location, effectively giving you a more ordered sequence. After sorting, picard then marks duplicates with a hex flag to make reading more efficient. Finally, all .bam files are converted to .mpileup format for use with VarScan.

Yi talked about how new of a field bioinformatics is, and consequently how many tools there are to do anything in the field. Converting to .mpileup is a prime example, because it doesn't actually change anything about the data, just converts it into .mpileup, still containing the same informtion. This is something NextFlow is trying to solve.

## 7. Call variants/filtering

Finally, we are finally getting to do what we set out to do! This process seems very easily pipelined and automated, and is very tedious to do by hand.

We now "call" indels and SNPs, outputted as .indel and .snp files, containing all the information about the variant calls for each.

Filtering is another important step. Some indels may cause changes to be read as SNPs, especially in the region around the indel site. Filtering takes out SNP calls that are too close to indel sites, and may not be reliable.

## 8. Comparison between genomes

These filtered reads can be exported as a CSV file, and can view a variety of data about the variant calls for each genome in a nice, human-readable format. Our final goal, though, is to compare mutations between parents and children. There is a nice script, find_unique_indels.py, that allows you to filter out unique indels and make tons of customized comparisons. This is really dependent on your needs, so there's no concrete steps after this.
