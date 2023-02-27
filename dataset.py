from collections import namedtuple

# Dataset root directory
DATASET_ROOT = './data'

Dataset = namedtuple('Dataset', ['name', 'version', 'src_file', 'labeled', 'mapped'])

ant13 = Dataset(
    'ant',
    '1.3',
    DATASET_ROOT + '/source file/ant-1.3/',
    DATASET_ROOT + '/labeled data/ant-1.3.csv',
    DATASET_ROOT + '/mapped data/ant-1.3.csv',
)

ant14 = Dataset(
    'ant',
    '1.4',
    DATASET_ROOT + '/source file/ant-1.4/',
    DATASET_ROOT + '/labeled data/ant-1.4.csv',
    DATASET_ROOT + '/mapped data/ant-1.4.csv',
)

ant15 = Dataset(
    'ant',
    '1.5',
    DATASET_ROOT + '/source file/ant-1.5/',
    DATASET_ROOT + '/labeled data/ant-1.5.csv',
    DATASET_ROOT + '/mapped data/ant-1.5.csv',
)

ant16 = Dataset(
    'ant',
    '1.6',
    DATASET_ROOT + '/source file/ant-1.6/',
    DATASET_ROOT + '/labeled data/ant-1.6.csv',
    DATASET_ROOT + '/mapped data/ant-1.6.csv',
)

ant17 = Dataset(
    'ant',
    '1.7',
    DATASET_ROOT + '/source file/ant-1.7/',
    DATASET_ROOT + '/labeled data/ant-1.7.csv',
    DATASET_ROOT + '/mapped data/ant-1.7.csv',
)

camel10 = Dataset(
    'camel',
    '1.0',
    DATASET_ROOT + '/source file/camel-1.0/',
    DATASET_ROOT + '/labeled data/camel-1.0.csv',
    DATASET_ROOT + '/mapped data/camel-1.0.csv',
)

camel12 = Dataset(
    'camel',
    '1.2',
    DATASET_ROOT + '/source file/camel-1.2/',
    DATASET_ROOT + '/labeled data/camel-1.2.csv',
    DATASET_ROOT + '/mapped data/camel-1.2.csv',
)

camel14 = Dataset(
    'camel',
    '1.4',
    DATASET_ROOT + '/source file/camel-1.4/',
    DATASET_ROOT + '/labeled data/camel-1.4.csv',
    DATASET_ROOT + '/mapped data/camel-1.4.csv',
)

camel16 = Dataset(
    'camel',
    '1.6',
    DATASET_ROOT + '/source file/camel-1.6/',
    DATASET_ROOT + '/labeled data/camel-1.6.csv',
    DATASET_ROOT + '/mapped data/camel-1.6.csv',
)

ivy11 = Dataset(
    'ivy',
    '1.1',
    DATASET_ROOT + '/source file/ivy-1.1/',
    DATASET_ROOT + '/labeled data/ivy-1.1.csv',
    DATASET_ROOT + '/mapped data/ivy-1.1.csv',
)

ivy14 = Dataset(
    'ivy',
    '1.4',
    DATASET_ROOT + '/source file/ivy-1.4/',
    DATASET_ROOT + '/labeled data/ivy-1.4.csv',
    DATASET_ROOT + '/mapped data/ivy-1.4.csv',
)

ivy20 = Dataset(
    'ivy',
    '2.0',
    DATASET_ROOT + '/source file/ivy-2.0/',
    DATASET_ROOT + '/labeled data/ivy-2.0.csv',
    DATASET_ROOT + '/mapped data/ivy-2.0.csv',
)

jedit32 = Dataset(
    'jedit',
    '3.2',
    DATASET_ROOT + '/source file/jedit-3.2/',
    DATASET_ROOT + '/labeled data/jedit-3.2.csv',
    DATASET_ROOT + '/mapped data/jedit-3.2.csv',
)

jedit40 = Dataset(
    'jedit',
    '4.0',
    DATASET_ROOT + '/source file/jedit-4.0/',
    DATASET_ROOT + '/labeled data/jedit-4.0.csv',
    DATASET_ROOT + '/mapped data/jedit-4.0.csv',
)

jedit41 = Dataset(
    'jedit',
    '4.1',
    DATASET_ROOT + '/source file/jedit-4.1/',
    DATASET_ROOT + '/labeled data/jedit-4.1.csv',
    DATASET_ROOT + '/mapped data/jedit-4.1.csv',
)

jedit42 = Dataset(
    'jedit',
    '4.2',
    DATASET_ROOT + '/source file/jedit-4.2/',
    DATASET_ROOT + '/labeled data/jedit-4.2.csv',
    DATASET_ROOT + '/mapped data/jedit-4.2.csv',
)

jedit43 = Dataset(
    'jedit',
    '4.3',
    DATASET_ROOT + '/source file/jedit-4.3/',
    DATASET_ROOT + '/labeled data/jedit-4.3.csv',
    DATASET_ROOT + '/mapped data/jedit-4.3.csv',
)

log4j10 = Dataset(
    'log4j',
    '1.0',
    DATASET_ROOT + '/source file/log4j-1.0/',
    DATASET_ROOT + '/labeled data/log4j-1.0.csv',
    DATASET_ROOT + '/mapped data/log4j-1.0.csv',
)

log4j11 = Dataset(
    'log4j',
    '1.1',
    DATASET_ROOT + '/source file/log4j-1.1/',
    DATASET_ROOT + '/labeled data/log4j-1.1.csv',
    DATASET_ROOT + '/mapped data/log4j-1.1.csv',
)

log4j12 = Dataset(
    'log4j',
    '1.2',
    DATASET_ROOT + '/source file/log4j-1.2/',
    DATASET_ROOT + '/labeled data/log4j-1.2.csv',
    DATASET_ROOT + '/mapped data/log4j-1.2.csv',
)

lucene20 = Dataset(
    'lucene',
    '2.0',
    DATASET_ROOT + '/source file/lucene-2.0/',
    DATASET_ROOT + '/labeled data/lucene-2.0.csv',
    DATASET_ROOT + '/mapped data/lucene-2.0.csv',
)

lucene22 = Dataset(
    'lucene',
    '2.2',
    DATASET_ROOT + '/source file/lucene-2.2/',
    DATASET_ROOT + '/labeled data/lucene-2.2.csv',
    DATASET_ROOT + '/mapped data/lucene-2.2.csv',
)

lucene24 = Dataset(
    'lucene',
    '2.4',
    DATASET_ROOT + '/source file/lucene-2.4/',
    DATASET_ROOT + '/labeled data/lucene-2.4.csv',
    DATASET_ROOT + '/mapped data/lucene-2.4.csv',
)

pbeans10 = Dataset(
    'pbeans',
    '1.0',
    DATASET_ROOT + '/source file/pbeans-1.0/',
    DATASET_ROOT + '/labeled data/pbeans-1.0.csv',
    DATASET_ROOT + '/mapped data/pbeans-1.0.csv',
)

pbeans20 = Dataset(
    'pbeans',
    '2.0',
    DATASET_ROOT + '/source file/pbeans-2.0/',
    DATASET_ROOT + '/labeled data/pbeans-2.0.csv',
    DATASET_ROOT + '/mapped data/pbeans-2.0.csv',
)

poi15 = Dataset(
    'poi',
    '1.5',
    DATASET_ROOT + '/source file/poi-1.5/',
    DATASET_ROOT + '/labeled data/poi-1.5.csv',
    DATASET_ROOT + '/mapped data/poi-1.5.csv',
)

poi20 = Dataset(
    'poi',
    '2.0',
    DATASET_ROOT + '/source file/poi-2.0/',
    DATASET_ROOT + '/labeled data/poi-2.0.csv',
    DATASET_ROOT + '/mapped data/poi-2.0.csv',
)

poi25 = Dataset(
    'poi',
    '2.5',
    DATASET_ROOT + '/source file/poi-2.5/',
    DATASET_ROOT + '/labeled data/poi-2.5.csv',
    DATASET_ROOT + '/mapped data/poi-2.5.csv',
)

poi30 = Dataset(
    'poi',
    '3.0',
    DATASET_ROOT + '/source file/poi-3.0/',
    DATASET_ROOT + '/labeled data/poi-3.0.csv',
    DATASET_ROOT + '/mapped data/poi-3.0.csv',
)

synapse10 = Dataset(
    'synapse',
    '1.0',
    DATASET_ROOT + '/source file/synapse-1.0/',
    DATASET_ROOT + '/labeled data/synapse-1.0.csv',
    DATASET_ROOT + '/mapped data/synapse-1.0.csv',
)

synapse11 = Dataset(
    'synapse',
    '1.1',
    DATASET_ROOT + '/source file/synapse-1.1/',
    DATASET_ROOT + '/labeled data/synapse-1.1.csv',
    DATASET_ROOT + '/mapped data/synapse-1.1.csv',
)

synapse12 = Dataset(
    'synapse',
    '1.2',
    DATASET_ROOT + '/source file/synapse-1.2',
    DATASET_ROOT + '/labeled data/synapse-1.2.csv',
    DATASET_ROOT + '/mapped data/synapse-1.2.csv',
)

velocity14 = Dataset(
    'velocity',
    '1.4',
    DATASET_ROOT + '/source file/velocity-1.4/',
    DATASET_ROOT + '/labeled data/velocity-1.4.csv',
    DATASET_ROOT + '/mapped data/velocity-1.4.csv',
)

velocity15 = Dataset(
    'velocity',
    '1.5',
    DATASET_ROOT + '/source file/velocity-1.5/',
    DATASET_ROOT + '/labeled data/velocity-1.5.csv',
    DATASET_ROOT + '/mapped data/velocity-1.5.csv',
)

velocity16 = Dataset(
    'velocity',
    '1.6',
    DATASET_ROOT + '/source file/velocity-1.6/',
    DATASET_ROOT + '/labeled data/velocity-1.6.csv',
    DATASET_ROOT + '/mapped data/velocity-1.6.csv',
)

xalan24 = Dataset(
    'xalan',
    '2.4',
    DATASET_ROOT + '/source file/xalan-2.4/',
    DATASET_ROOT + '/labeled data/xalan-2.4.csv',
    DATASET_ROOT + '/mapped data/xalan-2.4.csv',
)

xalan25 = Dataset(
    'xalan',
    '2.5',
    DATASET_ROOT + '/source file/xalan-2.5/',
    DATASET_ROOT + '/labeled data/xalan-2.5.csv',
    DATASET_ROOT + '/mapped data/xalan-2.5.csv',
)

xalan26 = Dataset(
    'xalan',
    '2.6',
    DATASET_ROOT + '/source file/xalan-2.6/',
    DATASET_ROOT + '/labeled data/xalan-2.6.csv',
    DATASET_ROOT + '/mapped data/xalan-2.6.csv',
)

xerces12 = Dataset(
    'xerces',
    '1.2',
    DATASET_ROOT + '/source file/xerces-1.2/',
    DATASET_ROOT + '/labeled data/xerces-1.2.csv',
    DATASET_ROOT + '/mapped data/xerces-1.2.csv',
)

xerces13 = Dataset(
    'xerces',
    '1.3',
    DATASET_ROOT + '/source file/xerces-1.3/',
    DATASET_ROOT + '/labeled data/xerces-1.3.csv',
    DATASET_ROOT + '/mapped data/xerces-1.3.csv',
)

xercesinit = Dataset(
    'xerces',
    'init',
    DATASET_ROOT + '/source file/xerces-init/',
    DATASET_ROOT + '/labeled data/xerces-init.csv',
    DATASET_ROOT + '/mapped data/xerces-init.csv',
)

DATASET = ant13
