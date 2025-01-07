"""SMAWK.py

Totally monotone matrix searching algorithms.

The offline algorithm in ConcaveMinima is from Agarwal, Klawe, Moran,
Shor, and Wilbur, Geometric applications of a matrix searching algorithm,
Algorithmica 2, pp. 195-208 (1987).

The online algorithm in OnlineConcaveMinima is from Galil and Park,
A linear time algorithm for concave one-dimensional dynamic programming,
manuscript, 1989, which simplifies earlier work on the same problem
by Wilbur (J. Algorithms 1988) and Eppstein (J. Algorithms 1990).

D. Eppstein, March 2002, significantly revised August 2005
"""

def ConcaveMinima(RowIndices, ColIndices, Matrix):
    """
    Search for the minimum value in each column of a matrix.
    The return value is a dictionary mapping ColIndices to pairs
    (value,rowindex). We break ties in favor of earlier rows.

    The matrix is defined implicitly as a function, passed
    as the third argument to this routine, where Matrix(i,j)
    gives the matrix value at row index i and column index j.
    The matrix must be concave, that is, satisfy the property
        Matrix(i,j) > Matrix(i',j) => Matrix(i,j') > Matrix(i',j')
    for every i<i' and j<j'; that is, in every submatrix of
    the input matrix, the positions of the column minima
    must be monotonically nondecreasing.

    The rows and columns of the matrix are labeled by the indices
    given in order by the first two arguments. In most applications,
    these arguments can simply be integer ranges.
    """

    # Base case of recursion
    if not ColIndices:
        return {}

    # Reduce phase: make number of rows at most equal to number of cols
    stack = []
    for r in RowIndices:
        while len(stack) >= 1 and \
                Matrix(stack[-1], ColIndices[len(stack) - 1]) \
                > Matrix(r, ColIndices[len(stack) - 1]):
            stack.pop()
        if len(stack) != len(ColIndices):
            stack.append(r)
    RowIndices = stack

    # Recursive call to search for every odd column
    minima = ConcaveMinima(RowIndices,
                           [ColIndices[i]
                               for i in range(1, len(ColIndices), 2)],
                           Matrix)

    # Go back and fill in the even rows
    r = 0
    for c in range(0, len(ColIndices), 2):
        col = ColIndices[c]
        row = RowIndices[r]
        if c == len(ColIndices) - 1:
            lastrow = RowIndices[-1]
        else:
            lastrow = minima[ColIndices[c + 1]][1]
        pair = (Matrix(row, col), row)
        while row != lastrow:
            r += 1
            row = RowIndices[r]
            pair = min(pair, (Matrix(row, col), row))
        minima[col] = pair

    return minima


class OnlineConcaveMinima:

    """
    Online concave minimization algorithm of Galil and Park.

    OnlineConcaveMinima(Matrix,initial) creates a sequence of pairs
    (self.value(j),self.index(j)), where
        self.value(0) = initial,
        self.value(j) = min { Matrix(i,j) | i < j } for j > 0,
    and where self.index(j) is the value of j that provides the minimum.
    Matrix(i,j) must be concave, in the same sense as for ConcaveMinima.

    We never call Matrix(i,j) until value(i) has already been computed,
    so that the Matrix function may examine previously computed values.
    Calling value(i) for an i that has not yet been computed forces
    the sequence to be continued until the desired index is reached.
    Calling iter(self) produces a sequence of (value,index) pairs.

    Matrix(i,j) should always return a value, rather than raising an
    exception, even for j larger than the range we expect to compute.
    If j is out of range, a suitable value to return that will not
    violate concavity is Matrix(i,j) = -i.  It will not work correctly
    to return a flag value such as None for large j, because the ties
    formed by the equalities among such flags may violate concavity.
    """

    def __init__(self, Matrix, initial):
        """Initialize a OnlineConcaveMinima object."""

        # State used by self.value(), self.index(), and iter(self)
        self._values = [initial]    # tentative solution values...
        self._indices = [None]      # ...and their indices
        self._finished = 0          # index of last non-tentative value

        # State used by the internal algorithm
        #
        # We allow self._values to be nonempty for indices > finished,
        # keeping invariant that
        # (1) self._values[i] = Matrix(self._indices[i], i),
        # (2) if the eventual correct value of self.index(i) < base,
        #     then self._values[i] is nonempty and correct.
        #
        # In addition, we keep a column index self._tentative, such that
        # (3) if i <= tentative, and the eventual correct value of
        #     self.index(i) <= finished, then self._values[i] is correct.
        #
        self._matrix = Matrix
        self._base = 0
        self._tentative = 0

    def __iter__(self):
        """Loop through (value,index) pairs."""
        i = 0
        while True:
            yield self.value(i), self.index(i)
            i += 1

    def value(self, j):
        """Return min { Matrix(i,j) | i < j }."""
        while self._finished < j:
            self._advance()
        return self._values[j]

    def index(self, j):
        """Return argmin { Matrix(i,j) | i < j }."""
        while self._finished < j:
            self._advance()
        return self._indices[j]

    def _advance(self):
        """Finish another value,index pair."""
        # First case: we have already advanced past the previous tentative
        # value.  We make a new tentative value by applying ConcaveMinima
        # to the largest square submatrix that fits under the base.
        i = self._finished + 1
        if i > self._tentative:
            rows = range(self._base, self._finished + 1)
            self._tentative = self._finished + len(rows)
            cols = range(self._finished + 1, self._tentative + 1)
            minima = ConcaveMinima(rows, cols, self._matrix)
            for col in cols:
                if col >= len(self._values):
                    self._values.append(minima[col][0])
                    self._indices.append(minima[col][1])
                elif minima[col][0] < self._values[col]:
                    self._values[col], self._indices[col] = minima[col]
            self._finished = i
            return

        # Second case: the new column minimum is on the diagonal.
        # All subsequent ones will be at least as low,
        # so we can clear out all our work from higher rows.
        # As in the fourth case, the loss of tentative is
        # amortized against the increase in base.
        diag = self._matrix(i - 1, i)
        if diag < self._values[i]:
            self._values[i] = diag
            self._indices[i] = self._base = i - 1
            self._tentative = self._finished = i
            return

        # Third case: row i-1 does not supply a column minimum in
        # any column up to tentative. We simply advance finished
        # while maintaining the invariant.
        prev_row = self._matrix(i - 1, self._tentative)
        tentative_value = self._values[self._tentative]
        if prev_row >= tentative_value:
            self._finished = i
            return

        # Fourth and final case: a new column minimum at self._tentative.
        # This allows us to make progress by incorporating rows
        # prior to finished into the base.  The base invariant holds
        # because these rows cannot supply any later column minima.
        # The work done when we last advanced tentative (and undone by
        # this step) can be amortized against the increase in base.
        self._base = i - 1
        self._tentative = self._finished = i
        return

class LineNumbers:
    def __init__(self):
        self.line_numbers = [0]

    def get(self, i, cost):
        while (pos := len(self.line_numbers)) < i + 1:
            line_number = 1 + self.get(cost.index(pos), cost)
            self.line_numbers.append(line_number)
        return self.line_numbers[i]

class Hyphens:
    def __init__(self):
        self.hyphens = [None]

    def hyphenate(self, word):
        import hyphen

        return [[word] if len(word) < 4 or ('=' in word) else hyphen.Hyphenator('en_US').syllables(word)]

    def get(self, i, cost):
        while (pos := len(self.line_numbers)) < i + 1:
            line_number = 1 + self.get(cost.index(pos), cost)
            self.line_numbers.append(line_number)
        return self.line_numbers[i]

class Fragment:
    def __init__(self, word, width, whitespace_width, penalty_width, hyphenator=None):
        self.word = word
        self.width = width
        self.whitespace_width = whitespace_width
        self.penalty_width = penalty_width
        self.hyphenator = hyphenator
        self._syllables = None

    def hyphenate(self):
        if self._syllables is None:
            self._syllables = [[self.word] if len(self.word) < 4 or ('=' in self.word) else self.hyphenator(self.word)]
        return self._syllables

def wrap(text,                 # string or unicode to be wrapped
         target=76,            # maximum length of a wrapped line
         measure=len,          # how to measure the length of a word
         overflow_penalty=1000,     # penalize long lines by overpen*(len-target)
         nlinepenalty=1000,    # penalize more lines than optimal
         short_last_line_fraction=10,    # penalize really short last line
         short_last_line_penalty=25,    # by this amount
         hyphen_penalty=25,   # penalize hyphenated words
         hyphenator=None,     # hyphenation function
         ):
    """Wrap the given text, returning a sequence of lines."""

    fragments = []
    for word in text.split():
        spacing = 1
        fragments.append(Fragment(word, measure(word), spacing, 0))

    fragments = []
    for word in text.split():
        for i, letter in enumerate(word):
            is_end = i == len(word) - 1
            fragments.append(Fragment(letter, measure(letter), 1 if is_end else 0, 0 if is_end else 1))

    widths = [0.0]
    width = 0.0
    for fragment in fragments:
        width += fragment.width + fragment.whitespace_width
        widths.append(width)

    line_numbers = LineNumbers()

    import numpy as np
    M = np.zeros((len(fragments) + 1, len(fragments) + 1))

    # Define penalty function for breaking on line words[i:j]
    # Below this definition we will set up cost[i] to be the
    # total penalty of all lines up to a break prior to word i.
    def penalty(i, j):
        if j > len(fragments):
            return -i    # concave flag for out of bounds

        line_number = line_numbers.get(i, cost)
        line_width = float(target) #/ (2 if line_number % 2 == 0 else 1)
        target_width = max(line_width, 1.0)

        #line_width = widths[j] - widths[i] - fragments[j - 1].whitespace_width + fragments[j - 1].penalty_width
        line_pre_width = widths[j - 1] - widths[i]
        line_width = line_pre_width + fragments[j - 1].width + fragments[j - 1].penalty_width

        c = cost.value(i) + nlinepenalty

        if line_width > target_width:
            if hyphenator is not None and line_pre_width < target_width:
                pass
            overflow = line_width - target_width
            c += overflow * overflow_penalty
        elif j < len(fragments):
            gap = target_width - line_width
            c += gap * gap
        elif i + 1 == j and line_width < target_width / short_last_line_fraction:
            c += short_last_line_penalty

        if fragments[j - 1].penalty_width > 0.0:
            c += hyphen_penalty

        M[i,j] = c
        return c

    # Apply concave minima algorithm and backtrack to form lines
    cost = OnlineConcaveMinima(penalty, 0)
    pos = len(fragments)
    lines = []
    while pos:
        breakpoint = cost.index(pos)
        print(pos, cost.value(pos) - cost.value(breakpoint), )
        line = []
        for i in range(breakpoint, pos):
            line.append(fragments[i].word)
            if not i + 1 == pos:
                line.append('_' * fragments[i].whitespace_width)
        if fragments[i].penalty_width > 0.0:
            line.append('-')
        lines.append(''.join(line))
        pos = breakpoint
    lines.reverse()

    return lines


if __name__ == "__main__":
    import re

    medium_long_text = \
        """Whether I shall turn out to be the hero of my own life, or whether that
        station will be held by anybody else, these pages must show. To begin my
        life with the beginning of my life, I record that I was born (as I have
        been informed and believe) on a Friday, at twelve o’clock at night.
        It was remarked that the clock began to strike, and I began to cry,
        simultaneously.

        In consideration of the day and hour of my birth, it was declared by
        the nurse, and by some sage women in the neighbourhood who had taken a
        lively interest in me several months before there was any possibility
        of our becoming personally acquainted, first, that I was destined to be
        unlucky in life; and secondly, that I was privileged to see ghosts and
        spirits; both these gifts inevitably attaching, as they believed, to
        all unlucky infants of either gender, born towards the small hours on a
        Friday night.

        I need say nothing here, on the first head, because nothing can show
        better than my history whether that prediction was verified or falsified
        by the result. On the second branch of the question, I will only remark,
        that unless I ran through that part of my inheritance while I was still
        a baby, I have not come into it yet. But I do not at all complain of
        having been kept out of this property; and if anybody else should be in
        the present enjoyment of it, he is heartily welcome to keep it.

        I was born with a caul, which was advertised for sale, in the
        newspapers, at the low price of fifteen guineas. Whether sea-going
        people were short of money about that time, or were short of faith and
        preferred cork jackets, I don’t know; all I know is, that there was but
        one solitary bidding, and that was from an attorney connected with the
        bill-broking business, who offered two pounds in cash, and the balance
        in sherry, but declined to be guaranteed from drowning on any higher
        bargain. Consequently the advertisement was withdrawn at a dead
        loss--for as to sherry, my poor dear mother’s own sherry was in the
        market then--and ten years afterwards, the caul was put up in a raffle
        down in our part of the country, to fifty members at half-a-crown a
        head, the winner to spend five shillings. I was present myself, and I
        remember to have felt quite uncomfortable and confused, at a part of
        myself being disposed of in that way. The caul was won, I recollect, by
        an old lady with a hand-basket, who, very reluctantly, produced from it
        the stipulated five shillings, all in halfpence, and twopence halfpenny
        short--as it took an immense time and a great waste of arithmetic, to
        endeavour without any effect to prove to her. It is a fact which will
        be long remembered as remarkable down there, that she was never drowned,
        but died triumphantly in bed, at ninety-two. I have understood that it
        was, to the last, her proudest boast, that she never had been on the
        water in her life, except upon a bridge; and that over her tea (to which
        she was extremely partial) she, to the last, expressed her indignation
        at the impiety of mariners and others, who had the presumption to go
        ‘meandering’ about the world. It was in vain to represent to her
        that some conveniences, tea perhaps included, resulted from this
        objectionable practice. She always returned, with greater emphasis and
        with an instinctive knowledge of the strength of her objection, ‘Let us
        have no meandering.’"""


    print('\n'.join(wrap(medium_long_text, 100)))