
import os
from pathlib import Path

from nz_data_challenge import evaluation

if __name__ == '__main__':

    accepted_dir = 'accepted'
    submit_top_dir = 'submission'
    public_dir = Path(os.environ['NZ_DATA_DIR']) / 'public'
    # truth_dir = Path(os.environ['NZ_DATA_DIR']) / 'reserved'
    truth_dir = 'reserved'
    submissions = evaluation.get_submissions(accepted_dir)
    results_top_dir = 'results'
    template_jinja_file = 'docs_results/_templates/entry_summary.rst.j2'
    suffix = 'wfd'

    evaluation.setup_submissions(submissions)
    evaluation.run_submissions(submissions, 'results', 'accepted')
    evaluation.evaluate_all_submissions(
        submissions,
        submit_top_dir,
        public_dir,
        truth_dir,
        results_top_dir,
        template_jinja_file,
        suffix,
    )
    evaluation.cleanup_submissions(submissions)
