from __future__ import annotations

from typing import List, Dict, Annotated, Union, Literal, TypeAlias, Self, Iterator, Tuple, Set, Iterable

from ck.pgm import State
from pydantic import BaseModel, Field, field_validator, PositiveInt, model_validator, NonNegativeFloat

from synthorus.model.datasource_spec import DatasourceSpec
from synthorus.model.defaults import DEFAULT_ID_FIELD, DEFAULT_COUNT_FIELD, DEFAULT_NAME, DEFAULT_AUTHOR, \
    DEFAULT_COMMENT, DEFAULT_RNG_N, DEFAULT_EPSILON, DEFAULT_MIN_CELL_SIZE, DEFAULT_MAX_ADD_ROWS
from synthorus.model.noiser_spec import NoiserSpec, NoiserSpecLaplace
from synthorus.simulator.condition_spec import ConditionSpec
from synthorus.utils.validate_inputs import validate_inputs


class ModelSpec(BaseModel):
    """
    A Model Spec is a serializable (JSON) representation of a complete synthetic data model.

    The purpose of a model spec is to provide a high-level description of a synthetic data
    system, which includes:
    * what are the random variables available in the system
    * what are the datasources for random variables
    * what cross-tables are to be generated from datasources to capture allowable statistics
    * what are the differential privacy parameters to use to protect privacy
    * what are the entities and fields for generating synthetic data
    * metadata for administrative purposes.
    """
    name: str = DEFAULT_NAME  # A model name
    author: str = DEFAULT_AUTHOR  # The model author
    comment: str = DEFAULT_COMMENT  # A model comment

    roots: List[str] = []
    """
    `roots` holds a list of directory paths (possibly relative to some externally
    specified working directory) that are searched to find datasource file.
    """

    rng_n: PositiveInt = DEFAULT_RNG_N
    """
    `rng_n` is a differential privacy parameter that controls the random number security level
    of the pseudorandom number generator. This is used in the class `SafeRandom`.
        4 is equivalent to AES128,
        5 is equivalent to AES192,
        6 is equivalent to AES256.
    For details of random variate production, see:
    Holohan, N., & Braghin, S. (2021, October). Secure random sampling in differential privacy.
    In European Symposium on Research in Computer Security (pp. 523-542). Springer, Cham.
    """

    pgm_crosstabs: Literal['clean', 'noisy'] = 'noisy'
    """
    `pgm_crosstabs` specifies whether entity PGMs are constructed from clean or noisy
    cross-tables. Normally this would be 'noisy', but 'clean' can be used for debug
    or demonstration purposes.
    """

    datasources: Dict[str, DatasourceSpec]  # All datasources available to this model.
    rvs: Dict[str, ModelRVSpec]  # All random variables of this model.
    crosstabs: Dict[str, ModelCrosstabSpec]  # All cross-tables.
    entities: Dict[str, ModelEntitySpec]  # All simulation entities, as `entity_name: entity_specification`.
    parameters: Dict[str, State] = {}  # Any model parameters, as `parameter_name: value`.

    @model_validator(mode='after')
    def validate_model_spec(self) -> Self:
        # Ensure no parameter name is the empty string
        for param_name in self.parameters.keys():
            if param_name == '':
                raise ValueError('parameter name cannot be the empty string')

        # Ensure each random variable name is not the empty string
        for name, rv_spec in self.rvs.items():
            if name == '':
                raise ValueError('random variable name cannot be the empty string')

        # Ensure each cross-table name is not the empty string
        for name, crosstab_spec in self.crosstabs.items():
            if name == '':
                raise ValueError('cross-table name cannot be the empty string')

        # Ensure each entity name is not the empty string
        for name, entity_spec in self.entities.items():
            if name == '':
                raise ValueError('entity name cannot be the empty string')

        # Ensure each cross-table rvs matches datasources.
        # Ensure that cross-table epsilon is non-zero if datasource sensitivity is non-zero.
        for crosstab_name, crosstab_spec in self.crosstabs.items():
            datasource_name: str = crosstab_spec.datasource
            datasource: DatasourceSpec = self.datasources[datasource_name]

            if datasource.sensitivity > 0 and crosstab_spec.epsilon == 0:
                raise ValueError(
                    f'cross-table {crosstab_name!r} has epsilon = 0 '
                    f'but datasource {datasource_name!r} has sensitivity = {datasource.sensitivity}'
                )

            rv_name: str
            for rv_name in crosstab_spec.rvs:
                if rv_name not in datasource.rvs:
                    raise ValueError(
                        f'cross-table {crosstab_name!r}'
                        f' has random variable {rv_name!r}'
                        f' that is not in datasource {datasource_name!r}'
                    )
            for rv_name in datasource.non_distribution_rvs:
                if rv_name not in crosstab_spec.rvs:
                    raise ValueError(
                        f'cross-table {crosstab_name!r}'
                        f' is missing non-distribution random variable {rv_name!r}'
                        f' from datasource {datasource_name!r}'
                    )

        return self


class ModelRVSpec(BaseModel):
    """
    A specification of a single random variable in a synthetic data model.
    """
    states: PositiveInt | List[State] | Literal['infer_distinct', 'infer_range', 'infer_max']
    ensure_none: bool = False


class ModelCrosstabSpec(BaseModel):
    rvs: List[str]
    datasource: str  # The datasource to use to create this cross-table
    epsilon: NonNegativeFloat = DEFAULT_EPSILON
    min_cell_size: NonNegativeFloat = DEFAULT_MIN_CELL_SIZE
    noiser: NoiserSpec = NoiserSpecLaplace(max_add_rows=DEFAULT_MAX_ADD_ROWS)

    @model_validator(mode='after')
    def validate_crosstab_spec(self) -> Self:
        # Ensure rvs are unique.
        rvs_set: Set[str] = set(self.rvs)
        if len(rvs_set) != len(self.rvs):
            raise ValueError('cross-table rvs must be unique within a cross-table')

        return self


class ForeignKeyField(BaseModel):
    """
    A ForeignKeyField records a foreign key that references the primary key of another entity.
    If entities x and y are in a one-to-many relationship, then entity y
    keeps a ForeignKeyField reference to entity x. Practically, this
    records the fact that entity y will have a foreign key to entity x.
    """
    foreign_key_field_name: str  # name of field holding ID for the dependency entity
    foreign_entity: str  # name of the entity being referenced


class ModelEntitySpec(BaseModel):
    id_field_name: str = DEFAULT_ID_FIELD  # name of field holding row ID for the entity
    count_field_name: str = DEFAULT_COUNT_FIELD  # name of field holding row count for the entity
    foreign_key_fields: List[ForeignKeyField] = []
    fields: Dict[str, ModelFieldSpec] = {}
    cardinality: List[ConditionSpec] = []

    def sampled_fields(self) -> Iterator[Tuple[str, ModelFieldSpecSample]]:
        """
        Iterate over fields that are sampled random variables.

        Returns:
            An iterator over (field_name, model_field_spec_sample) pairs.
        """
        for field_name, field in self.fields.items():
            if isinstance(field, ModelFieldSpecSample):
                yield field_name, field

    @model_validator(mode='after')
    def validate_model(self) -> Self:
        self.check_fields()
        return self

    def check_fields(self) -> None:
        """
        Raises:
            ValueError: if the fields are not unique
        """
        check_fields(
            self.id_field_name,
            self.count_field_name,
            self.foreign_key_fields,
            self.fields.keys(),
        )


def check_fields(
        id_field_name: str,
        count_field_name: str,
        foreign_key_fields: List[ForeignKeyField],
        other_field_names: Iterable[str],
) -> None:
    """
    Check field names and foreign entity names are unique and valid names.

    Raises:
        ValueError: if the fields are inconsistent
    """
    # foreign field names must be unique
    foreign_key_field_names: Set[str] = {dependency.foreign_key_field_name for dependency in foreign_key_fields}
    if len(foreign_key_field_names) != len(foreign_key_fields):
        raise ValueError(f'foreign key field names are not unique')

    # foreign entities must be unique
    foreign_entity_names: Set[str] = {dependency.foreign_entity for dependency in foreign_key_fields}
    if len(foreign_entity_names) != len(foreign_key_fields):
        raise ValueError(f'foreign entity names are not unique')

    other_field_names: Set[str] = set(other_field_names)

    # foreign key field names must be different to the explicit file names
    for foreign_field_name in foreign_key_field_names:
        if foreign_field_name in other_field_names:
            raise ValueError(f'foreign field name cannot be the same as an explicit field: {foreign_field_name!r}')

    # id and count fields must be different to each other
    if count_field_name == id_field_name:
        raise ValueError(f'count and id field names must be different')

    # id and count fields must be different to foreign fields and explicit fields
    all_fields: Set[str] = foreign_key_field_names.union(other_field_names)
    if id_field_name in all_fields:
        raise ValueError(
            f'id field name cannot be the same as an explicit or foreign field: {id_field_name!r}'
        )
    if count_field_name in all_fields:
        raise ValueError(
            f'count field name cannot be the same as an explicit or foreign field: {id_field_name!r}'
        )

    # no field can be the empty string
    all_fields.add(id_field_name)
    all_fields.add(count_field_name)
    for field_name in all_fields:
        if field_name == '':
            raise ValueError('field name cannot be the empty string')


class ModelFieldSpecSample(BaseModel):
    """
    A ModelFieldSpec specifying the field should be sampled.
    """
    type: Literal['sample'] = 'sample'  # for JSON round trip
    rv_name: str  # name of the random variable to sample.


class ModelFieldSpecSum(BaseModel):
    type: Literal['sum'] = 'sum'  # for JSON round trip
    initial_value: State = 0  # initial field state (default is 0)
    sum: List[str]  # list of source fields to sum
    offset: State = 0  # optional offset value to sum


class ModelFieldSpecFunction(BaseModel):
    type: Literal['function'] = 'function'  # for JSON round trip
    initial_value: State = None  # initial field state (default is None)
    inputs: List[str]
    function: str  # Python expression representing the body of a function

    # noinspection PyNestedDecorators
    @field_validator('inputs', mode='before')
    @classmethod
    def validate_inputs(cls, inputs: List[str]) -> List[str]:
        return validate_inputs(inputs)


ModelFieldSpec: TypeAlias = Annotated[
    Union[
        ModelFieldSpecSample,
        ModelFieldSpecSum,
        ModelFieldSpecFunction,
    ],
    Field(discriminator='type')
]
