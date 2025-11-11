# Contributing

Thankyou for considering to contribute to the Synthorus project.

## Issues

Issues are very valuable to this project.

  - Ideas are a valuable source of contributions others can make.
  - Problems show where this project is lacking.
  - With a question you show where contributors can improve the user experience.

Issues can be logged at https://github.com/ropeless/synthorus/issues.

Thank you for creating them.

## Pull Requests

Pull requests are a great way to get your ideas into this repository.

When deciding whether to merge a pull request the following
are considered.

  - Does the request fix an actual bug or satisfy an actual need within the vision of the project?
  - Is the request in keeping with the design principles and conventions of the project?
  - Is code well-designed, clear and efficient?
  - Are there adequate unit tests covering code?
  - Is code documentation clear, stating the semantics, intent and limitations of the code?
  - Is language used well, without spelling mistakes or poor English?
  - Is the request respectful and free of offensive material?

## Design Principles and Conventions

This project aims to exhibit excellent design and coding practices.

- Each code component (package, module, function, variable, ...) should have a single responsibility
  that is clear to users and maintainers of the component.

- Code components should be loosely coupled, with dependencies between components minimised
  and clearly understandable. Code covariances should be avoided, especially between broadly separated
  code components. (A code covariance is where a change at one place in the code requires a
  change at a different place.)

- Avoid redundancy by implementing reusable code components.

- Avoid over-design and over-engineering that complicates code.

- Any code at risk of bugs should have unit tests. Pull requests that purport to fix a bug should
  have a unit test that fails when the bug is present and passes when the fix is applied.

- Don't add comments that are immediately obvious from the code. An exception
  is to include a function docstring as the body code may not be available to someone
  reading the documentation. No need to include function documentation that is
  immediately obvious given the function declaration.

- Always include the semantics and intention of code in code comments. Don't merely explain
  the implementation. Make caveats, limitations and important details clear.

- Use conventional design patterns to make code understandable.

- Prefer to use Python type hints.

- Use Google docstring style.

- Use standard English for comments and identifiers.
