# Anonymized Dataset — Year Folders

This folder contains the raw, anonymized transaction data used throughout this project,
split into yearly `.csv` files (e.g., `2024_full_anonymus_v2_final.csv`).

## A note on the year labels (2027–2029)

Some of the yearly files in this folder are labeled with years that have not yet occurred
at the time of dataset release (e.g., 2027, 2028, 2029).

**This is intentional and is a direct consequence of the anonymization process, not an
indication that this data was collected in the future or synthetically generated.**

As part of the anonymization pipeline, all order dates in the original dataset were shifted by a fixed temporal offset. This offset was applied to prevent the original collection period from being identifiable, while fully preserving the sequential and seasonal structure of the data (e.g., day-of-week patterns, month-to-month seasonality, and inter-purchase intervals are all unaffected by the shift).

As a result, some transactions that were originally recorded in the past now fall under calendar years that appear to be in the future relative to the dataset's release date. The six years of data span a fixed real-world period; only the calendar labels have been shifted, not the underlying temporal structure or duration.

For more details on the anonymization pipeline, see Section 2.1 of the accompanying paper.