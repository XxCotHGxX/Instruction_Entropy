## Work description

Extract relevant data points from each provided report, particularly structured findings, and output them into a specified format.

### Data Extraction

From each PDF report, parse and extract structured information based on the following format:
- Finding Number
- Title (e.g. Requirements Violation, Documentation Mismatch)
- Status
- Severity
- Impact
- Likelihood
- Description
- Recommendation
- Repository URL
- Commit ID

If some of these fields are not present in a given report, leave them out. If other fields are present in a report, include those as well (e.g., "Type" or "Target").

## Provided material

Folder with PDFs to extract information from: `inputs/Audit_Work_PDFs`

## Deliverables

Final extracted data in structured text format, containing all findings (one file per report)