%function [ordered_data, new_order, PCA_results] = BLIND_v1(DATA,dyn_q,PC_dim,var_scale,use_entropy, plot_results)
function [ordered_data, new_order, PCA_results] = BLIND_v1(DATA,dyn_q,PC_dim,var_scale,use_entropy, plot_results)
% CONSTANTS
tspo_pop  = 200;
tspo_iter = 2e4;
% input validation and sanity checks
nargs = 5;
for k = nargin:nargs-1
    switch k
        case 0
            error('Invalid input - data is missing')
        case 1
            dyn_q = 0.95;
        case 2
            PC_dim = 2;
        case 3
            var_scale = 0;
        case 4
            use_entropy = 0;
        otherwise
    end
end
if (dyn_q <= 0) || (dyn_q >= 1)
    error('Invalid input - "dyn_q" must be in the range (0,1)')
end
var_scale = logical(var_scale);
m = size(DATA,2); %#ok<*NODEF>
% filter dynamic genes
dyn_genes = 1:size(DATA,1);
if (size(DATA,1) > 20)
    dyn_genes = gene_filter_d(DATA,dyn_q);
end
% Handle log scaled data
l_DATA = log10(1+DATA);
[~,SCORE, latent, ~] = princomp(l_DATA(dyn_genes,:)');
size(SCORE);
A = SCORE(:,1:PC_dim);
A_scaled = scale_PCA(A);
% scale PCs according to explained variance
if var_scale
    precentage_variance = latent./sum(latent);
    for d=1:PC_dim
        v = precentage_variance(d);
        A_scaled(:,d) = A_scaled(:,d)*v;
    end
end
if (plot_results)
    figure;
    scatter(A_scaled(:,1),A_scaled(:,2),60,'k','filled','MarkerEdgeColor','black','LineWidth',1.3);
end
% traveling salesman
dmat = squareform(pdist(A_scaled));
new_order = tspo_ga(A_scaled,dmat,tspo_pop,tspo_iter,0,0);
% entropy
entropy=calc_entropy(l_DATA(dyn_genes,new_order));
p = polyfit(1:m, entropy,1);
color_vector = jet(size(A_scaled,1));
if use_entropy
    if p(1)<0
        new_order = new_order(end:-1:1);
        entropy = entropy(end:-1:1);
    end
    if (plot_results)
        figure;
        samples = 1:m;
        scatter(samples,entropy,60,color_vector,'filled','MarkerEdgeColor','black','LineWidth',1.3);
    end
end
if (plot_results)
    color_vector = jet(size(A_scaled,1));
    figure;
    hold on;
    for i = 1:(m-1)
        plot(A_scaled(new_order(i:(i+1)),1),A_scaled(new_order(i:(i+1)),2),'-','Color',color_vector(i,:));
    end
    scatter(A_scaled(:,1),A_scaled(:,2),60,'k','filled','MarkerEdgeColor','black','LineWidth',1.3);
end
% save results
ordered_data = DATA(:,new_order);
PCA_results = A_scaled;
end

%% help functions
function [inds] = gene_filter_d(DATA,q)
gene_data = DATA';
temp = sort(gene_data);
MIN = temp(2,:);
MAX = temp(end-1,:);
dynamic = MAX-MIN;
thresh = quantile(dynamic,q);
inds = find(dynamic>thresh);
end

function [A_scaled] = scale_PCA(A)
A_scaled = (A - repmat(min(A),size(A,1),1))./repmat((max(A) - min(A)),size(A,1),1);
Precision = 100;
A_scaled=ceil(A_scaled*Precision);
A_scaled=A_scaled-min(A_scaled(:))+15;
end

function entropy= calc_entropy (DATA)
normed = DATA / norm(DATA, 1);
plogp = normed.*log(normed);
plogp(isnan(plogp)) = 0;
entropy =  sum(-plogp);
end

function varargout = tspo_ga(xy,dmat,popSize,numIter,showProg,showResult)
% Process Inputs and Initialize Defaults
nargs = 6;
for k = nargin:nargs-1
    switch k
        case 0
            xy = 10*rand(50,2);
        case 1
            N = size(xy,1);
            a = meshgrid(1:N);
            dmat = reshape(sqrt(sum((xy(a,:)-xy(a',:)).^2,2)),N,N);
        case 2
            popSize = 100;
        case 3
            numIter = 1e4;
        case 4
            showProg = 1;
        case 5
            showResult = 1;
        otherwise
    end
end

% Verify Inputs
[N,dims] = size(xy);
[nr,nc] = size(dmat);
if N ~= nr || N ~= nc
    error('Invalid XY or DMAT inputs!')
end
n = N;

% Sanity Checks
popSize = 4*ceil(popSize/4);
numIter = max(1,round(real(numIter(1))));
showProg = logical(showProg(1));
showResult = logical(showResult(1));

% Initialize the Population
pop = zeros(popSize,n);
pop(1,:) = (1:n);
for k = 2:popSize
    pop(k,:) = randperm(n);
end

% Run the GA
globalMin = Inf;
totalDist = zeros(1,popSize);
distHistory = zeros(1,numIter);
tmpPop = zeros(4,n);
newPop = zeros(popSize,n);
if showProg
    pfig = figure('Name','TSPO_GA | Current Best Solution','Numbertitle','off');
    f = 1;
end
for iter = 1:numIter
    % Evaluate Each Population Member (Calculate Total Distance)
    for p = 1:popSize
        d = 0; % Open Path
        for k = 2:n
            d = d + dmat(pop(p,k-1),pop(p,k));
        end
        totalDist(p) = d;
    end
    
    % Find the Best Route in the Population
    [minDist,index] = min(totalDist);
    distHistory(iter) = minDist;
    if minDist < globalMin
        globalMin = minDist;
        optRoute = pop(index,:);
        if showProg
            % Plot the Best Route
            figure(pfig);
            M(f) = getframe(gcf);
            f = f+1;
            if dims > 2, plot3(xy(optRoute,1),xy(optRoute,2),xy(optRoute,3),'r.-');
            else plot(xy(optRoute,1),xy(optRoute,2),'r.-'); end
            title(sprintf('Total Distance = %1.4f, Iteration = %d',minDist,iter));
        end
    end
    
    % Genetic Algorithm Operators
    randomOrder = randperm(popSize);
    for p = 4:4:popSize
        rtes = pop(randomOrder(p-3:p),:);
        dists = totalDist(randomOrder(p-3:p));
        [ignore,idx] = min(dists); %#ok
        bestOf4Route = rtes(idx,:);
        routeInsertionPoints = sort(ceil(n*rand(1,2)));
        I = routeInsertionPoints(1);
        J = routeInsertionPoints(2);
        for k = 1:4 % Mutate the Best to get Three New Routes
            tmpPop(k,:) = bestOf4Route;
            switch k
                case 2 % Flip
                    tmpPop(k,I:J) = tmpPop(k,J:-1:I);
                case 3 % Swap
                    tmpPop(k,[I J]) = tmpPop(k,[J I]);
                case 4 % Slide
                    tmpPop(k,I:J) = tmpPop(k,[I+1:J I]);
                otherwise % Do Nothing
            end
        end
        newPop(p-3:p,:) = tmpPop;
    end
    pop = newPop;
end

if showProg
    %     movie2avi(M,'genetic.avi');
end
if showResult
    % Plots the GA Results
    figure('Name','TSPO_GA | Results','Numbertitle','off');
    subplot(2,2,1);
    pclr = ~get(0,'DefaultAxesColor');
    if dims > 2, plot3(xy(:,1),xy(:,2),xy(:,3),'.','Color',pclr);
    else plot(xy(:,1),xy(:,2),'.','Color',pclr); end
    title('City Locations');
    subplot(2,2,2);
    imagesc(dmat(optRoute,optRoute));
    title('Distance Matrix');
    subplot(2,2,3);
    if dims > 2, plot3(xy(optRoute,1),xy(optRoute,2),xy(optRoute,3),'r.-');
    else plot(xy(optRoute,1),xy(optRoute,2),'r.-'); end
    title(sprintf('Total Distance = %1.4f',minDist));
    subplot(2,2,4);
    plot(distHistory,'b','LineWidth',2);
    title('Best Solution History');
    set(gca,'XLim',[0 numIter+1],'YLim',[0 1.1*max([1 distHistory])]);
end

% Return Outputs
if nargout
    varargout{1} = optRoute;
    varargout{2} = minDist;
end
end