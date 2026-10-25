//! The packed admission state one generation decides entry and retirement on.

use crate::session_generation::admission::Admissions;

#[test]
fn closing_refuses_entry_without_changing_the_admitted_count() {
    let admissions = Admissions::open();
    assert!(admissions.admit());
    assert!(admissions.admit());
    assert_eq!(admissions.admitted(), 2);
    assert!(admissions.is_open());

    assert!(
        admissions.close(),
        "the first close reports the open-to-closed transition"
    );
    assert!(!admissions.close(), "a second close is not a transition");
    assert!(!admissions.is_open());
    assert!(
        !admissions.admit(),
        "a closed generation admits no new call"
    );
    assert_eq!(
        admissions.admitted(),
        2,
        "a refused call leaves no entry in the drained total"
    );

    admissions.release();
    admissions.release();
    assert_eq!(admissions.admitted(), 0);
    assert!(!admissions.is_open(), "releasing does not reopen");
}
