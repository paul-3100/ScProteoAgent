# Licence status: pending rights-holder confirmation

No software licence has been chosen for this repository, so **no open-source licence is
granted** and the source is published under the default terms of the rights holder until a
licence file is added. `CITATION.cff` therefore carries no `license` field at all: the 1.2.0
schema makes it optional, and a placeholder such as `LicenseRef-Proprietary` would read as a
decision that has not been taken. The field is added when the licence is chosen.

What is still open (author/rights-holder decision):

1. Which licence applies to the **software** in this repository (engine, scripts, reproduction
   code, tests, examples). Copyright remains with the authors and their institutions.
2. Which licence applies to the **data archive** that accompanies the manuscript. Data terms
   are recorded separately from the software licence and are not implied by it.
3. Whether the third-party resources listed in `THIRD_PARTY_NOTICES.md` may be redistributed.
   Until that is confirmed, they are *not* part of any release artefact.

Until the licence is added:

* do not describe this repository as open source, and do not present a licence badge;
* keep redistribution decisions with the rights holder;
* keep this file in the repository so the pending state stays visible.

When the licence is chosen, add the licence text as `LICENSE` (or `LICENSE.md`) beside this file,
add the `license` field with the SPDX identifier to `CITATION.cff`, and record the decision in the
release manifest. This file may then be deleted, or kept as a record of the decision.

## The two routes currently on the table

Both are drafts prepared for the rights holder; neither is in force, and no licence file was
added. The supporting material (quoted policy text, the draft terms, the standard
open-source alternative and a side-by-side decision sheet) is delivered with the release
support package rather than in this repository.

1. **A research / non-commercial licence.** Free for research, teaching, reproducing published
   results and non-commercial modification and redistribution; prior written authorisation for
   company-internal R&D, paid analysis services and commercial product integration. Not an
   OSI-approved licence, so it departs from the Nature Portfolio *recommendation* to use one,
   while still satisfying the mandatory obligation to disclose the licence and its
   restrictions in the Code Availability statement.
2. **GPL-3.0-only**, which is OSI-approved and permits commercial use, in exchange for the
   copyleft distribution obligations. AGPL-3.0 is the variant to consider if the concern is a
   modified copy being run as a closed network service.

The identity of the granting entity, and confirmation that every contributor agrees, still
have to be settled before either can be adopted.
